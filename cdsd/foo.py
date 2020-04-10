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
from data import get_data_transforms, CDSDDataset, SAMPLE_RATE
from models import construct_separator, construct_classifier
from torchaudio.functional import istft, magphase, spectrogram
from utils import get_torch_window_fn
from loudness import compute_dbfs


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

    batch_size = train_config["training"]["batch_size"]

    subset = 'test'
    dataset = CDSDDataset(root_data_dir,
                          subset=subset,
                          transform=input_transform,
                          load_separation_data=True)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, pin_memory=True,
                            num_workers=num_data_workers)
    num_batches = len(dataloader)

    subset_results_path = os.path.join(output_dir, "separation_results_{}.csv".format(subset))

    for batch_idx, batch in enumerate(dataloader):
        curr_batch_size = batch['labels'].size()[0]
        if save_audio:
            recon_audio_dir = os.path.join(output_dir, "reconstructed_audio")
        recon_masks_dir = os.path.join(output_dir, "reconstructed_masks")

        for label_idx, label in enumerate(train_dataset.labels):
            if save_audio:
                for f_idx in range(curr_batch_size):
                    file = dataset.files[batch_idx * batch_size + f_idx]
                    recon_out_path = os.path.join(recon_audio_dir, "{}_{}_recon.wav".format(file, label))
                    print(recon_out_path)

        del batch


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
