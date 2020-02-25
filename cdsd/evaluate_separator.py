import argparse
import json
import os
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import get_data_transforms, CDSDDataset
from models import construct_separator
from torchaudio.functional import istft, magphase, spectrogram
from utils import get_torch_window_fn


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

    parser.add_argument('-n', '--num-data-workers',
                        type=int, default=1,
                        help='Number of workers used for data loading.')

    return parser.parse_args(args)


def evaluate(root_data_dir, train_config, output_dir=None, num_data_workers=1):
    # Create output directory
    output_dir = output_dir or train_config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    input_transform = get_data_transforms(train_config)

    # get STFT parameters:
    params = {}
    for transform_config in train_config["input_transforms"]:
        if transform_config["name"] != 'Spectrogram':
            continue
        params = transform_config["parameters"]
        break
    spec_params = {
        "n_fft": params.get("n_fft", 400),
        "power": params.get("power", 2.0),
        "normalized": params.get("normalized", False),
        "window_fn": get_torch_window_fn(params.get("window_fn", "hann_window")),
        "wkwargs": params.get("wkwargs", {})
    }
    spec_params["win_length"] = params.get("win_length") or (spec_params["n_fft"] // 2 + 1)
    spec_params["hop_length"] = params.get("hop_length") or (spec_params["win_length"] // 2)

    train_dataset = CDSDDataset(root_data_dir,
                                subset='train',
                                transform=input_transform)
    separator = construct_separator(train_config,
                                    dataset=train_dataset,
                                    require_init=True)

    for subset in ('train', 'valid', 'test'):
        print('====== Evaluating subset "{}" ======'.format(subset))
        dataset = CDSDDataset(root_data_dir,
                              subset=subset,
                              transform=input_transform,
                              load_separator_data=True)
        dataloader = DataLoader(dataset, batch_size=train_config["training"]["batch_size"],
                                shuffle=False, pin_memory=True,
                                num_workers=num_data_workers)
        num_batches = len(dataloader)

        subset_results = {label + "_input_sisdr": [] for label in dataset.labels}
        subset_results.update({label + "_sisdr_improvement": [] for label in dataset.labels})
        subset_results["filenames"] = list(dataset.files) # Assuming that dataloader preserves order

        subset_results_path = os.path.join(output_dir, "separation_results_{}.csv".format(subset))

        for batch in tqdm(dataloader, total=num_batches):
            x = batch["audio_data"]
            labels = batch["labels"]
            mixture_waveforms = batch["waveform"]

            mixture_maggram, mixture_phasegram = magphase(spectrogram(mixture_waveforms,
                                    window=spec_params["window_fn"](
                                        window_length=spec_params["win_length"], **spec_params["wkwargs"]),
                                    n_fft=spec_params["n_fft"],
                                    hop_length=spec_params["hop_length"],
                                    win_length=spec_params["win_length"],
                                    power=None,
                                    normalized=spec_params["normalized"]))

            cos_phasegram = torch.cos(mixture_phasegram)
            sin_phasegram = torch.sin(mixture_phasegram)

            masks = separator(x)

            for idx, label in enumerate(train_dataset.labels):
                source_waveforms = batch[label + "_waveform"]

                # Reconstruct source audio
                recon_source_maggram = mixture_maggram * masks[..., idx]
                recon_source_stft = torch.zeros(mixture_maggram.size() + (2,))
                recon_source_stft[..., 0] = recon_source_maggram * cos_phasegram
                recon_source_stft[..., 1] = recon_source_maggram * sin_phasegram
                recon_source_waveforms = istft(
                    recon_source_stft,
                    window=spec_params["window_fn"](
                        window_length=spec_params["win_length"], **spec_params["wkwargs"]),
                    n_fft=spec_params["n_fft"],
                    hop_length=spec_params["hop_length"],
                    win_length=spec_params["win_length"],
                    normalized=spec_params["normalized"],
                    onesided=True,
                    center=True,
                    pad_mode="reflect")

                # Compute SI-SDR with mixture as estimated source
                input_sisdr = compute_sisdr(mixture_waveforms, source_waveforms)
                # Compute SI-SDR improvement using reconstructed sources
                sisdr_imp = compute_sisdr(recon_source_waveforms, source_waveforms) - input_sisdr

                # If label is not present, then set SDR to NaN
                sisdr_imp[torch.logical_not(labels[..., idx].bool())] = float('nan')

                subset_results[label + "_input_sisdr"] += input_sisdr.tolist()
                subset_results[label + "_sisdr_improvement"] += sisdr_imp.tolist()

        # Save results as
        subset_df = pd.DataFrame(subset_results)
        subset_df.to_csv(subset_results_path)

        print("Saved results to {}".format(subset_results_path))


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])

    # Load training config
    with open(args.train_config_path, 'r') as f:
        train_config = json.load(f)

    evaluate(root_data_dir=args.root_data_dir,
             train_config=train_config,
             output_dir=args.output_dir,
             num_data_workers=args.num_data_workers)
