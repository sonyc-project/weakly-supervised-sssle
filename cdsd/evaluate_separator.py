import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import torch
import torchaudio
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import get_data_transforms, CDSDDataset, SAMPLE_RATE, get_spec_params, get_mel_params, get_mel_loss_params
from models import construct_separator, construct_classifier
from torchaudio.functional import magphase, lowpass_biquad
from transforms import istft, spectrogram
from loudness import compute_dbfs_spec


EPS = 1e-8


def sqnorm(signal):
    return torch.pow(torch.norm(signal, p=2, dim=-1, keepdim=True), 2)


def rowdot(s1, s2):
    return torch.sum(s1 * s2, dim=-1, keepdim=True)


def get_references(batch, label_list):
    refs = []
    for label in label_list:
        # TODO: Double check that it is okay that matrix is not full rank.
        # Don't think it should matter
        label_waveform = torch.squeeze(batch["{}_waveform".format(label)], dim=1)[..., None]
        refs.append(label_waveform)

    if "background_waveform" in batch:
        label_waveform = torch.squeeze(batch["background_waveform"])[..., None]
        refs.append(label_waveform)

    return torch.cat(refs, dim=-1)


def compute_source_separation_metrics(estimated, original, references, include_lpf=False):
    # Adapted from https://github.com/sigsep/bsseval/issues/3#issuecomment-494995846
    if estimated.dim() == 3:
        estimated = torch.squeeze(estimated, dim=1)
    if original.dim() == 3:
        original = torch.squeeze(original, dim=1)

    _estimated = estimated
    _original = original
    _references = references

    if include_lpf:
        lpf_options = (True, False)
    else:
        lpf_options = (False,)

    metrics = {}
    for lpf in lpf_options:
        prefix = ""
        if lpf:
            # Apply lowpass filter to look at SI-SDR specific to low frequencies
            cutoff = 1000
            original = lowpass_biquad(original,
                                      sample_rate=SAMPLE_RATE,
                                      cutoff_freq=cutoff)
            estimated = lowpass_biquad(estimated,
                                       sample_rate=SAMPLE_RATE,
                                       cutoff_freq=cutoff)
            references = lowpass_biquad(references.transpose(-2, -1).contiguous(),
                                        sample_rate=SAMPLE_RATE,
                                        cutoff_freq=cutoff).transpose(-2, -1)
            prefix = "lpf"
        else:
            original = _original
            estimated = _estimated
            references = _references

        Rss = torch.matmul(references.transpose(-2, -1), references)
        Rss = Rss + EPS * torch.eye(Rss.shape[-1])[None, ...].to(Rss.device)

        for scale_invariant in (True, False):
            if scale_invariant:
                a = rowdot(estimated, original) / (sqnorm(original) + EPS)
            else:
                a = 1

            e_true = a * original
            e_res = estimated - e_true

            Sss = sqnorm(e_true)
            Snn = sqnorm(e_res)

            sdr = 10 * torch.log10(Sss / (Snn + EPS) + EPS)
            Rsr = torch.matmul(references.transpose(-2, -1), e_res[..., None])
            # NOTE: Torch's argument order is opposite of numpy's

            b, _ = torch.solve(Rsr, Rss)

            e_interf = torch.matmul(references, b)[..., 0]
            e_artif = e_res - e_interf

            sir = 10 * torch.log10(Sss / (sqnorm(e_interf) + EPS) + EPS)
            sar = 10 * torch.log10(Sss / (sqnorm(e_artif) + EPS) + EPS)

            sdr = sdr.squeeze_(dim=-1)
            sir = sir.squeeze_(dim=-1)
            sar = sar.squeeze_(dim=-1)

            if scale_invariant:
                metrics[prefix + 'sisdr'] = sdr
                metrics[prefix + 'sisir'] = sir
                metrics[prefix + 'sisar'] = sar
                metrics[prefix + 'sdsdr'] = 10 * torch.log10(Sss / (sqnorm(original - estimated) + EPS) + EPS)
            else:
                metrics[prefix + 'sdr'] = sdr
                metrics[prefix + 'sir'] = sir
                metrics[prefix + 'sar'] = sar

    return metrics


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

    parser.add_argument('--save-masks',
                        action='store_true',
                        help='If true, save the estimated masks')

    parser.add_argument('--include-lpf',
                        action='store_true',
                        help='If true, also compute source separation metrics with lowpass filtering')

    parser.add_argument('--checkpoint',
                        type=str, default='best', choices=('best', 'latest', 'earlystopping'),
                        help='Type of model checkpoint to load.')

    parser.add_argument('-n', '--num-data-workers',
                        type=int, default=1,
                        help='Number of workers used for data loading.')

    parser.add_argument('--verbose',
                        action="store_true",
                        help='If selected, print verbose output.')

    return parser.parse_args(args)


def evaluate(root_data_dir, train_config, output_dir=None, num_data_workers=1, save_audio=False, save_masks=False, include_lpf=False, checkpoint='best', verbose=False):
    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Memory hack: https://discuss.pytorch.org/t/solved-pytorch-conv2d-consumes-more-gpu-memory-than-tensorflow/28998/2
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

    # Create output directory
    output_dir = output_dir or train_config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # get STFT parameters:
    spec_params = get_spec_params(train_config)
    mel_params = get_mel_params(train_config) or get_mel_loss_params(train_config)

    # Set up data loaders
    input_transform = get_data_transforms(train_config)
    train_dataset = CDSDDataset(root_data_dir,
                                subset='train',
                                transform=input_transform)

    batch_size = train_config["training"]["batch_size"]
    label_mode = train_config["training"]["label_mode"]

    # Set up models
    separator = construct_separator(train_config,
                                    dataset=train_dataset,
                                    require_init=True,
                                    trainable=False,
                                    device=device,
                                    checkpoint=checkpoint)
    classifier = construct_classifier(train_config,
                                      dataset=train_dataset,
                                      label_mode=label_mode,
                                      require_init=True,
                                      trainable=False,
                                      device=device,
                                      checkpoint=checkpoint)
    if torch.cuda.device_count() > 1:
        print("Using {} GPUs for evaluation.".format(torch.cuda.device_count()))
        separator = nn.DataParallel(separator)
        classifier = nn.DataParallel(classifier)
    separator.to(device)
    classifier.to(device)
    separator.eval()
    classifier.eval()

    # Set up label downsampling
    input_num_frames = train_dataset.get_num_frames()
    output_num_frames = classifier.get_num_frames(input_num_frames)
    assert input_num_frames >= output_num_frames
    if label_mode == "frame" and input_num_frames > output_num_frames:
        # Set up max pool
        pool_size = input_num_frames // output_num_frames
        label_maxpool = nn.MaxPool1d(pool_size)
        label_maxpool.to(device)
        label_maxpool.eval()
    else:
        label_maxpool = None

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
            source_sep_metric_names = ["sisdr", "sisir", "sisar", "sdr", "sir", "sar", "sdsdr"]
            for metric_name in source_sep_metric_names:
                subset_results.update({label + "_input_{}".format(metric_name): []
                                       for label in dataset.labels})
                subset_results.update({label + "_recon_{}".format(metric_name): []
                                       for label in dataset.labels})
                if include_lpf:
                    subset_results.update({label + "_input_lpf{}".format(metric_name): []
                                           for label in dataset.labels})
                    subset_results.update({label + "_recon_lpf{}".format(metric_name): []
                                           for label in dataset.labels})

            subset_results.update({"mixture_dbfs": []})
            subset_results.update({"isolated_" + label + "_dbfs": [] for label in dataset.labels})
            subset_results.update({"reconstructed_" + label + "_dbfs": [] for label in dataset.labels})

            subset_results.update({label + "_presence_gt": [] for label in dataset.labels})
            subset_results.update({label + "_presence_frame_gt": [] for label in dataset.labels})
            subset_results.update({"mixture_pred_" + label: [] for label in dataset.labels})
            subset_results.update({"isolated_" + gt_label + "_pred_" + pred_label: [] for gt_label in dataset.labels for pred_label in dataset.labels})
            subset_results.update({"reconstructed_" + gt_label + "_pred_" + pred_label: [] for gt_label in dataset.labels for pred_label in dataset.labels})

            subset_results_path = os.path.join(output_dir, "separation_results_{}_{}.csv".format(checkpoint, subset))

            for batch_idx, batch in tqdm(enumerate(dataloader), total=num_batches, disable=(not verbose)):
                x = batch["audio_data"].to(device)
                clip_labels = batch["clip_labels"].to(device)
                frame_labels = batch["frame_labels"].to(device)
                if label_mode == "frame" and label_maxpool is not None:
                    frame_labels = label_maxpool(frame_labels.transpose(1, 2)).transpose(1, 2)
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
                    normalized=spec_params["normalized"],
                    window_scaling=spec_params["window_scaling"]), power=1.0)

                # Sanity check
                # JTC: Removed for now due to GPU/CPU discrepancies...
                #assert torch.allclose(x, mixture_maggram, atol=1e-5)

                cos_phasegram = torch.cos(mixture_phasegram)
                sin_phasegram = torch.sin(mixture_phasegram)

                # Run separator on mixture to obtain masks
                masks = separator(x)

                # Run classifier on mixture for later analysis
                mixture_cls_pred = classifier(x)

                if save_audio:
                    recon_audio_dir = os.path.join(output_dir, "{}_reconstructed_audio".format(checkpoint))
                    os.makedirs(recon_audio_dir, exist_ok=True)
                if save_masks:
                    recon_masks_dir = os.path.join(output_dir, "{}_reconstructed_masks".format(checkpoint))
                    os.makedirs(recon_masks_dir, exist_ok=True)

                # Compute dBFS for the mixture
                subset_results["mixture_dbfs"] += compute_dbfs_spec(x, SAMPLE_RATE, spec_params, mel_params=mel_params, device=device).squeeze().tolist()

                for label_idx, label in enumerate(train_dataset.labels):
                    source_waveforms = batch[label + "_waveform"].to(device)
                    source_maggram = batch[label + "_transformed"].to(device)

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
                        window_scaling=spec_params["window_scaling"],
                        onesided=True,
                        center=True,
                        pad_mode="reflect")

                    # Compute dBFS for the isolated and reconstructed sources
                    subset_results["isolated_" + label + "_dbfs"] += compute_dbfs_spec(source_maggram, SAMPLE_RATE, spec_params, mel_params=mel_params, device=device).squeeze().tolist()
                    subset_results["reconstructed_" + label + "_dbfs"] += compute_dbfs_spec(recon_source_maggram, SAMPLE_RATE, spec_params, mel_params=mel_params, device=device).squeeze().tolist()

                    reference_waveforms = get_references(batch, train_dataset.labels).to(device)
                    # Compute source separation metrics with mixture as estimated source
                    input_ss_metrics = compute_source_separation_metrics(mixture_waveforms,
                                                                         source_waveforms,
                                                                         reference_waveforms,
                                                                         include_lpf=include_lpf)
                    # Compute source separation metrics with reconstructed sources
                    recon_ss_metrics = compute_source_separation_metrics(recon_source_waveforms,
                                                                         source_waveforms,
                                                                         reference_waveforms,
                                                                         include_lpf=include_lpf)

                    # Save source separation metrics
                    for metric_name in source_sep_metric_names:
                        subset_results["{}_input_{}".format(label, metric_name)] += input_ss_metrics[metric_name].tolist()
                        subset_results["{}_recon_{}".format(label, metric_name)] += recon_ss_metrics[metric_name].tolist()
                        if include_lpf:
                            subset_results["{}_input_lpf{}".format(label, metric_name)] += input_ss_metrics["lpf" + metric_name].tolist()
                            subset_results["{}_recon_lpf{}".format(label, metric_name)] += recon_ss_metrics["lpf" + metric_name].tolist()

                    # Run classifier on isolated source for later analysis
                    source_cls_pred = classifier(source_maggram)
                    for pred_idx, pred_label in enumerate(train_dataset.labels):
                        subset_results["isolated_" + label + "_pred_" + pred_label] += source_cls_pred[..., pred_idx].tolist()

                    # Run classifier on reconstructed source for later analysis
                    source_cls_pred = classifier(recon_source_maggram)
                    for pred_idx, pred_label in enumerate(train_dataset.labels):
                        subset_results["reconstructed_" + label + "_pred_" + pred_label] += source_cls_pred[..., pred_idx].tolist()

                    # Save ground truth labels and mixture classification results (since we're already iterating through labels)
                    subset_results[label + "_presence_gt"] += clip_labels[:, label_idx].tolist()
                    subset_results[label + "_presence_frame_gt"] += frame_labels[..., label_idx].tolist()
                    subset_results["mixture_pred_" + label] += mixture_cls_pred[..., label_idx].tolist()

                    if save_audio:
                        for f_idx in range(recon_source_waveforms.size()[0]):
                            file = dataset.files[batch_idx * batch_size + f_idx]
                            # Save as MP3 to save space
                            recon_out_path = os.path.join(recon_audio_dir, "{}_{}_recon.mp3".format(file, label))
                            torchaudio.save(recon_out_path,
                                            recon_source_waveforms[f_idx, :].cpu(),
                                            sample_rate=SAMPLE_RATE)
                            assert os.path.exists(recon_out_path)

                    if save_masks:
                        # Save ideal and predicted masks for analysis and debugging
                        for f_idx in range(masks.size()[0]):
                            file = dataset.files[batch_idx * batch_size + f_idx]
                            recon_out_path = os.path.join(recon_masks_dir, "{}_{}_recon.npz".format(file, label))

                            recon_mask = masks[f_idx, ..., label_idx].cpu().numpy()
                            ideal_mask = source_ideal_ratio_mask[f_idx, ...].cpu().numpy()
                            file_mixture_maggram = mixture_maggram[f_idx, ...].cpu().numpy()

                            np.savez_compressed(recon_out_path,
                                                recon_mask=recon_mask,
                                                ideal_mask=ideal_mask,
                                                mixture_spectrogram=file_mixture_maggram)
                            assert os.path.exists(recon_out_path)

                del x, clip_labels, frame_labels, mixture_waveforms, mixture_maggram, \
                    mixture_phasegram, cos_phasegram, sin_phasegram, \
                    source_waveforms, source_maggram, recon_source_maggram, \
                    recon_source_stft, recon_source_waveforms, input_ss_metrics, \
                    recon_ss_metrics, source_cls_pred, source_ideal_ratio_mask, batch

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
             save_audio=args.save_audio,
             save_masks=args.save_masks,
             include_lpf=args.include_lpf,
             checkpoint=args.checkpoint,
             verbose=args.verbose)
