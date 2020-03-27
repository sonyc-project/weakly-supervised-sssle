import argparse
import json
import os
import sys
import torch
import torchaudio
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import get_data_transforms, CDSDDataset, SAMPLE_RATE
from models import construct_separator, construct_classifier
from torchaudio.functional import istft, magphase, spectrogram
from utils import get_torch_window_fn
from loudness import compute_dbfs


EPS = 1e-8


def sqnorm(signal):
    return torch.pow(torch.norm(signal, p=2, dim=-1, keepdim=True), 2)


def rowdot(s1, s2):
    return torch.sum(s1 * s2, dim=-1, keepdim=True)


def compute_sisdr(estimated, original):
    # Adapted from https://github.com/wangkenpu/Conv-TasNet-PyTorch/blob/master/utils/evaluate/si_sdr_torch.py
    if estimated.dim() == 3:
        estimated = torch.squeeze(estimated)
    if original.dim() == 3:
        original = torch.squeeze(original)

    signal = rowdot(estimated, original) * original / (sqnorm(original) + EPS)
    dist = estimated - signal
    sdr = 10 * torch.log10(sqnorm(signal) / (sqnorm(dist) + EPS) + EPS)
    return sdr.squeeze_(dim=-1)


def parse_arguments(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('root_data_dir',
                        type=str,
                        help='Path to dataset directory')

    parser.add_argument('train_config_path',
                        type=str,
                        help='Path to training configuration JSON file')

    parser.add_argument('--output-dir',
                        type=str,
                        help='Path where outputs will be saved. Defaults to the one specified in the train configuration file.')

    parser.add_argument('--save-audio',
                        action='store_true',
                        help='If true, save the reconstructed audio')

    parser.add_argument('-n', '--num-data-workers',
                        type=int, default=1,
                        help='Number of workers used for data loading.')

    return parser.parse_args(args)


def evaluate(root_data_dir, train_config, output_dir=None, num_data_workers=1, save_audio=False):
    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Memory hack: https://discuss.pytorch.org/t/solved-pytorch-conv2d-consumes-more-gpu-memory-than-tensorflow/28998/2
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

    # Create output directory
    output_dir = output_dir or train_config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # get STFT parameters:
    params = {}
    for transform_config in train_config["input_transforms"]:
        if transform_config["name"] != 'Spectrogram':
            continue
        params = transform_config["parameters"]
        break
    spec_params = {
        "pad": params.get("pad", 0),
        "n_fft": params.get("n_fft", 400),
        "power": params.get("power", 2.0),
        "normalized": params.get("normalized", False),
        "window_fn": get_torch_window_fn(params.get("window_fn", "hann_window")),
        "wkwargs": params.get("wkwargs", {})
    }
    spec_params["win_length"] = params.get("win_length") or spec_params["n_fft"]
    spec_params["hop_length"] = params.get("hop_length") or (spec_params["win_length"] // 2)

    # Set up data loaders
    input_transform = get_data_transforms(train_config)
    train_dataset = CDSDDataset(root_data_dir,
                                subset='train',
                                transform=input_transform)

    # Set up models
    separator = construct_separator(train_config,
                                    dataset=train_dataset,
                                    require_init=True,
                                    trainable=False)
    classifier = construct_classifier(train_config,
                                      dataset=train_dataset,
                                      require_init=True,
                                      trainable=False)
    if torch.cuda.device_count() > 1:
        print("Using {} GPUs for evaluation.".format(torch.cuda.device_count()))
        separator = nn.DataParallel(separator)
        classifier = nn.DataParallel(classifier)
    separator.to(device)
    classifier.to(device)
    separator.eval()
    classifier.eval()

    batch_size = train_config["training"]["batch_size"]

    with torch.no_grad():
        for subset in ('train', 'valid', 'test'):
            print('====== Evaluating subset "{}" ======'.format(subset))
            dataset = CDSDDataset(root_data_dir,
                                  subset=subset,
                                  transform=input_transform,
                                  load_separation_data=True)
            dataloader = DataLoader(dataset, batch_size=batch_size,
                                    shuffle=False, pin_memory=True,
                                    num_workers=num_data_workers)
            num_batches = len(dataloader)

            # Initialize results lists
            subset_results = {"filenames": list(dataset.files)} # Assuming that dataloader preserves order
            subset_results.update({label + "_input_sisdr": [] for label in dataset.labels})
            subset_results.update({label + "_sisdr_improvement": [] for label in dataset.labels})

            subset_results.update({"mixture_dbfs": []})
            subset_results.update({"isolated_" + label + "_dbfs": [] for label in dataset.labels})
            subset_results.update({"reconstructed_" + label + "_dbfs": [] for label in dataset.labels})

            subset_results.update({label + "_input_sisdr": [] for label in dataset.labels})
            subset_results.update({label + "_sisdr_improvement": [] for label in dataset.labels})

            subset_results.update({label + "_presence_gt": [] for label in dataset.labels})
            subset_results.update({"mixture_pred_" + label: [] for label in dataset.labels})
            subset_results.update({"isolated_" + gt_label + "_pred_" + pred_label: [] for gt_label in dataset.labels for pred_label in dataset.labels})
            subset_results.update({"reconstructed_" + gt_label + "_pred_" + pred_label: [] for gt_label in dataset.labels for pred_label in dataset.labels})

            subset_results_path = os.path.join(output_dir, "separation_results_{}.csv".format(subset))

            for batch_idx, batch in tqdm(enumerate(dataloader), total=num_batches):
                x = batch["audio_data"].to(device)
                labels = batch["labels"].to(device)
                mixture_waveforms = batch["mixture_waveform"].to(device)

                # Compute cosine and sine of phase spectrogram for reconstruction
                mixture_maggram, mixture_phasegram = magphase(spectrogram(
                    mixture_waveforms,
                    pad=spec_params["pad"],
                    window=spec_params["window_fn"](
                        window_length=spec_params["win_length"],
                        **spec_params["wkwargs"]).to(device),
                    n_fft=spec_params["n_fft"],
                    hop_length=spec_params["hop_length"],
                    win_length=spec_params["win_length"],
                    power=None,
                    normalized=spec_params["normalized"]), power=1.0)

                # Sanity check
                assert torch.allclose(x, mixture_maggram, atol=1e-7)

                cos_phasegram = torch.cos(mixture_phasegram)
                sin_phasegram = torch.sin(mixture_phasegram)

                # Run separator on mixture to obtain masks
                masks = separator(x)

                # Run classifier on mixture for later analysis
                mixture_cls_pred = classifier(x)

                if save_audio:
                    recon_audio_dir = os.path.join(output_dir, "reconstructed_audio")
                    os.makedirs(recon_audio_dir, exist_ok=True)
                recon_masks_dir = os.path.join(output_dir, "reconstructed_masks")
                os.makedirs(recon_masks_dir, exist_ok=True)

                # Compute dBFS for the mixture
                subset_results["mixture_dbfs"] += compute_dbfs(mixture_waveforms, SAMPLE_RATE, device=device).squeeze().tolist()

                for label_idx, label in enumerate(train_dataset.labels):
                    source_waveforms = batch[label + "_waveform"].to(device)
                    source_maggram = batch[label + "_transformed"].to(device)

                    # Compute cosine and sine of phase spectrogram for reconstruction
                    source_maggram, source_phasegram = magphase(spectrogram(
                        source_waveforms,
                        pad=spec_params["pad"],
                        window=spec_params["window_fn"](
                            window_length=spec_params["win_length"],
                            **spec_params["wkwargs"]).to(device),
                        n_fft=spec_params["n_fft"],
                        hop_length=spec_params["hop_length"],
                        win_length=spec_params["win_length"],
                        power=None,
                        normalized=spec_params["normalized"]), power=1.0)

                    # Compute IRM for debugging
                    source_ideal_ratio_mask = source_maggram / mixture_maggram

                    # Reconstruct source audio
                    recon_source_maggram = x * masks[..., label_idx]
                    recon_source_stft = torch.zeros(x.size() + (2,)).to(device)
                    recon_source_stft[..., 0] = recon_source_maggram * cos_phasegram
                    recon_source_stft[..., 1] = recon_source_maggram * sin_phasegram
                    recon_source_waveforms = istft(
                        recon_source_stft,
                        window=spec_params["window_fn"](
                            window_length=spec_params["win_length"], **spec_params["wkwargs"]).to(device),
                        n_fft=spec_params["n_fft"],
                        hop_length=spec_params["hop_length"],
                        win_length=spec_params["win_length"],
                        normalized=spec_params["normalized"],
                        onesided=True,
                        center=True,
                        pad_mode="reflect")

                    # Compute dBFS for the isolated and reconstructed sources
                    subset_results["isolated_" + label + "_dbfs"] += compute_dbfs(source_waveforms, SAMPLE_RATE, device=device).squeeze().tolist()
                    subset_results["reconstructed_" + label + "_dbfs"] += compute_dbfs(recon_source_waveforms, SAMPLE_RATE, device=device).squeeze().tolist()

                    # Compute SI-SDR with mixture as estimated source
                    input_sisdr = compute_sisdr(mixture_waveforms, source_waveforms)
                    # Compute SI-SDR improvement using reconstructed sources
                    sisdr_imp = compute_sisdr(recon_source_waveforms, source_waveforms) - input_sisdr

                    # If label is not present, then set SDR to NaN
                    # Removing for now, we can always do this masking later...
                    # sisdr_imp[torch.logical_not(labels[..., idx].bool())] = float('nan')

                    # Run classifier on isolated source for later analysis
                    source_cls_pred = classifier(source_maggram)
                    for pred_idx, pred_label in enumerate(train_dataset.labels):
                        subset_results["isolated_" + label + "_pred_" + pred_label] += source_cls_pred[:, pred_idx].tolist()

                    # Run classifier on reconstructed source for later analysis
                    source_cls_pred = classifier(recon_source_maggram)
                    for pred_idx, pred_label in enumerate(train_dataset.labels):
                        subset_results["reconstructed_" + label + "_pred_" + pred_label] += source_cls_pred[:, pred_idx].tolist()

                    # Save source separation metrics
                    subset_results[label + "_input_sisdr"] += input_sisdr.tolist()
                    subset_results[label + "_sisdr_improvement"] += sisdr_imp.tolist()

                    # Save ground truth labels and mixture classification results (since we're already iterating through labels)
                    subset_results[label + "_presence_gt"] += labels[:, label_idx].tolist()
                    subset_results["mixture_pred_" + label] += mixture_cls_pred[:, label_idx].tolist()

                    if save_audio:
                        for f_idx in range(recon_source_waveforms.size()[0]):
                            file = dataset.files[batch_idx * batch_size + f_idx]
                            # Save as MP3 to save space
                            recon_out_path = os.path.join(recon_audio_dir, "{}_{}_recon.mp3".format(file, label))
                            torchaudio.save(recon_out_path,
                                            recon_source_waveforms[f_idx, :].cpu(),
                                            sample_rate=SAMPLE_RATE)
                            assert os.path.exists(recon_out_path)

                    # Save ideal and predicted masks for analysis and debugging
                    for f_idx in range(masks.size()[0]):
                        file = dataset.files[batch_idx * batch_size + f_idx]
                        recon_out_path = os.path.join(recon_masks_dir, "{}_{}_recon.npy.gz".format(file, label))

                        recon_mask = masks[f_idx, ..., label_idx].cpu().numpy()
                        ideal_mask = source_ideal_ratio_mask[f_idx, ...].cpu().numpy()
                        file_mixture_maggram = mixture_maggram[f_idx, ...].cpu().numpy()

                        np.savez_compressed(recon_out_path,
                                            recon_mask=recon_mask,
                                            ideal_mask=ideal_mask,
                                            mixture_spectrogram=file_mixture_maggram)
                        assert os.path.exists(recon_out_path)

                del x, labels, mixture_waveforms, mixture_maggram, \
                    mixture_phasegram, cos_phasegram, sin_phasegram, \
                    source_waveforms, source_maggram, recon_source_maggram, \
                    recon_source_stft, recon_source_waveforms, input_sisdr, \
                    sisdr_imp, source_cls_pred, source_ideal_ratio_mask, batch
                torch.cuda.empty_cache()

        # Save results as CSV
        subset_df = pd.DataFrame(subset_results)
        subset_df.to_csv(subset_results_path)

        assert os.path.exists(subset_results_path)
        print("Saved results to {}".format(subset_results_path))


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])

    # Load training config
    with open(args.train_config_path, 'r') as f:
        train_config = json.load(f)

    evaluate(root_data_dir=args.root_data_dir,
             train_config=train_config,
             output_dir=args.output_dir,
             num_data_workers=args.num_data_workers,
             save_audio=args.save_audio)
