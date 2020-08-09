import argparse
import os
import json
import random
import sys
import shutil
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader
from torchaudio.transforms import MelScale
from tqdm import tqdm
from data import get_data_transforms, CDSDDataset, SAMPLE_RATE
from models import construct_separator, construct_classifier
from losses import get_normalization_factor
from utils import get_optimizer
from logs import FSSSHistoryLogger


def parse_arguments(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('root_data_dir',
                        type=str,
                        help='Path to dataset directory')

    parser.add_argument('train_config_path',
                        type=str,
                        help='Path to training configuration JSON file')

    parser.add_argument('output_dir',
                        type=str,
                        help='Path where outputs will be saved')

    parser.add_argument('-n', '--num-data-workers',
                        type=int, default=1,
                        help='Number of workers used for data loading.')

    parser.add_argument('-c', '--checkpoint-interval',
                        type=int, default=10,
                        help='Number of epochs in between checkpoints')

    parser.add_argument('-d', '--num-debug-examples',
                        type=int, default=5,
                        help='Number of debug examples to save')

    parser.add_argument('-s', '--save-debug-interval',
                        type=int, default=5,
                        help='Number of epochs in between saving debug examples')

    parser.add_argument('--random-state',
                        type=int, default=12345678,
                        help='Random state for reproducability')

    parser.add_argument('--verbose',
                        action="store_true",
                        help='If selected, print verbose output.')

    return parser.parse_args(args)


def train(root_data_dir, train_config, output_dir, num_data_workers=1,
          checkpoint_interval=10, num_debug_examples=5, save_debug_interval=5,
          random_state=12345678, verbose=False):
    # Set random seed
    torch.manual_seed(random_state)
    random.seed(random_state)
    np.random.seed(random_state)

    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Memory hack: https://discuss.pytorch.org/t/solved-pytorch-conv2d-consumes-more-gpu-memory-than-tensorflow/28998/2
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set up data loaders
    input_transform = get_data_transforms(train_config)
    batch_size = train_config["training"]["batch_size"]

    train_dataset = CDSDDataset(root_data_dir,
                                subset='train',
                                transform=input_transform,
                                load_separation_data=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, pin_memory=True,
                                  num_workers=num_data_workers)
    num_train_batches = len(train_dataloader)

    valid_dataset = CDSDDataset(root_data_dir,
                                subset='valid',
                                transform=input_transform,
                                load_separation_data=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                                  shuffle=True, pin_memory=True,
                                  num_workers=num_data_workers)
    num_valid_batches = len(valid_dataloader)

    # Set up models
    separator = construct_separator(train_config, dataset=train_dataset,
                                    device=device)
    multi_gpu = False
    if torch.cuda.device_count() > 1:
        num_devices = torch.cuda.device_count()
        multi_gpu = True
        print("Using {} GPUs for training.".format(num_devices))
        separator = nn.DataParallel(separator)

    separator.to(device)

    class_prior_weighting = train_config["training"].get("class_prior_weighting", False)
    energy_masking = train_config["losses"]["separation"].get("energy_masking", False)
    spectrum = train_config["losses"]["separation"].get("spectrum", False)
    mel_scale = train_config["losses"]["separation"].get("mel_scale", False)
    mel_params = train_config["losses"]["separation"].get("mel_params", {})

    mel_tf = None
    if mel_scale:
        mel_tf = MelScale(sample_rate=SAMPLE_RATE, **mel_params).to(device)

    patience = train_config["training"].get("early_stopping_patience", 5)
    early_stopping_terminate = train_config["training"].get("early_stopping_terminate", False)

    # JTC: Should we still provide params with requires_grad=False here?
    optimizer = get_optimizer(separator.parameters(), train_config)

    # Set up history logging
    history_path = os.path.join(output_dir, "history.csv")
    history_logger = FSSSHistoryLogger(history_path)

    # Set up checkpoint paths
    separator_best_ckpt_path = os.path.join(output_dir, "separator_best.pt")
    separator_latest_ckpt_path = os.path.join(output_dir, "separator_latest.pt")
    separator_earlystopping_ckpt_path = os.path.join(output_dir, "separator_earlystopping.pt")
    optimizer_latest_ckpt_path = os.path.join(output_dir, "optimizer_latest.pt")

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        save_config = deepcopy(train_config)
        save_config["root_data_dir"] = root_data_dir
        save_config["output_dir"] = output_dir
        save_config["num_data_workers"] = num_data_workers
        save_config["separator"]["best_path"] = separator_best_ckpt_path
        save_config["separator"]["latest_path"] = separator_latest_ckpt_path
        save_config["separator"]["earlystopping_path"] = separator_earlystopping_ckpt_path
        save_config["random_state"] = random_state
        json.dump(save_config, f)

    epoch = 0
    train_masks_save = None
    train_idxs_save = None
    valid_masks_save = None
    valid_idxs_save = None

    early_stopping_wait = 0
    early_stopping_flag = False
    num_epochs = train_config["training"]["num_epochs"]
    for epoch in range(num_epochs):
        print("=============== Epoch {}/{} =============== ".format(epoch + 1, num_epochs))
        accum_train_loss = 0.0
        accum_valid_loss = 0.0

        train_masks_save = None
        train_idxs_save = None
        valid_masks_save = None
        valid_idxs_save = None

        print(" **** Training ****")
        # Set models to train mode
        separator.train()
        torch.cuda.empty_cache()
        for batch_idx, batch in enumerate(tqdm(train_dataloader, total=num_train_batches, disable=(not verbose))):
            x = batch["audio_data"].to(device)
            frame_labels = batch["frame_labels"].to(device)
            energy_mask = batch["energy_mask"].to(device)
            curr_batch_size = x.size()[0]
            norm_factor = get_normalization_factor(x, energy_mask,
                                                   energy_masking=energy_masking,
                                                   spectrum=spectrum)

            if epoch == 0 and batch_idx == 0:
                print("Input size: {}".format(x.size()))

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass through separator
            masks = separator(x)

            # Compute mixture loss for separator
            train_loss = None
            src_spec = None
            src_spec_diff = None
            src_spec_diff_flat = None
            src_loss = None

            for label_idx, label in enumerate(train_dataset.labels):
                if class_prior_weighting:
                    cls_frame_labels = frame_labels[..., label_idx]
                    p = train_dataset.class_frame_priors[label_idx]

                    weight = torch.zeros_like(cls_frame_labels, dtype=x.dtype, device=device)
                    weight[cls_frame_labels.bool()] = 1.0 / p
                    weight[(-cls_frame_labels + 1).bool()] = 1.0 / (1 - p)
                    weight = weight[:, None, None, :]
                    del cls_frame_labels
                else:
                    weight = torch.ones(1, dtype=x.dtype, device=device)

                mask = masks[..., label_idx]
                x_masked = (x * mask).to(device)

                # Compute loss
                src_spec = batch[label + "_transformed"].to(device)

                # Optionally apply mel scale
                if mel_scale:
                    src_spec_diff = (mel_tf(src_spec) - mel_tf(x_masked)) * weight
                else:
                    src_spec_diff = (src_spec - x_masked) * weight

                if energy_masking:
                    src_spec_diff = src_spec_diff * energy_mask[:, None, None, :]

                if spectrum:
                    src_spec_diff = src_spec_diff.view(curr_batch_size, -1)
                else:
                    # Sum over time and channels
                    src_spec_diff = src_spec_diff.sum(dim=-1).sum(dim=1)

                src_loss = torch.norm(src_spec_diff, p=1, dim=1) / norm_factor
                src_loss = src_loss.mean()

                # Accumulate loss for each source
                if train_loss is None:
                    train_loss = src_loss
                else:
                    train_loss = train_loss + src_loss

            # Backprop
            train_loss.backward()
            optimizer.step()

            # Accumulate loss for epoch
            accum_train_loss += train_loss.item()

            # Save debug outputs (for first N training examples)
            if epoch % save_debug_interval == 0 and batch_idx == 0:
                train_masks_save = masks[:num_debug_examples].cpu().detach().numpy()
                train_idxs_save = batch['index'][:num_debug_examples].cpu().numpy()

            # Cleanup
            del x, masks, energy_mask, batch, mask, x_masked, train_loss, weight, \
                norm_factor, src_spec, src_spec_diff, src_spec_diff_flat, src_loss
            torch.cuda.empty_cache()

        # Evaluate on validation set
        print(" **** Validation ****")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(valid_dataloader, total=num_valid_batches, disable=(not verbose))):
                x = batch["audio_data"].to(device)
                frame_labels = batch["frame_labels"].to(device)
                energy_mask = batch["energy_mask"].to(device)
                curr_batch_size = x.size()[0]
                norm_factor = get_normalization_factor(x, energy_mask,
                                                       energy_masking=energy_masking)
                # Set models to eval mode
                separator.eval()

                # Forward pass through separator
                masks = separator(x)

                # Compute mixture loss for separator
                valid_loss = None
                src_spec = None
                src_spec_diff = None
                src_spec_diff_flat = None
                src_loss = None

                for label_idx, label in enumerate(valid_dataset.labels):
                    if class_prior_weighting:
                        cls_frame_labels = frame_labels[..., label_idx]
                        p = train_dataset.class_frame_priors[label_idx]

                        weight = torch.zeros_like(cls_frame_labels, dtype=x.dtype, device=device)
                        weight[cls_frame_labels.bool()] = 1.0 / p
                        weight[(-cls_frame_labels + 1).bool()] = 1.0 / (1 - p)
                        weight = weight[:, None, None, :]
                        del cls_frame_labels
                    else:
                        weight = torch.ones(1, dtype=x.dtype, device=device)

                    mask = masks[..., label_idx]
                    x_masked = (x * mask).to(device)

                    # Compute loss
                    src_spec = batch[label + "_transformed"].to(device)
                    src_spec_diff = (src_spec - x_masked) * weight
                    if energy_masking:
                        src_spec_diff = src_spec_diff * energy_mask[:, None, None, :]
                    src_spec_diff_flat = src_spec_diff.view(curr_batch_size, -1)
                    src_loss = torch.norm(src_spec_diff_flat, p=1, dim=1) / norm_factor
                    src_loss = src_loss.mean()

                    # Accumulate loss for each source
                    if valid_loss is None:
                        valid_loss = src_loss
                    else:
                        valid_loss = valid_loss + src_loss

                # Accumulate loss for epoch
                accum_valid_loss += valid_loss.item()

                # Save debug outputs (for first N validing examples)
                if epoch % save_debug_interval == 0 and batch_idx == 0:
                    valid_masks_save = masks[:num_debug_examples].cpu().detach().numpy()
                    valid_idxs_save = batch['index'][:num_debug_examples].cpu().numpy()

                # Cleanup
                del x, masks, energy_mask, batch, mask, x_masked, valid_loss, weight, \
                    norm_factor, src_spec, src_spec_diff, src_spec_diff_flat, src_loss
                torch.cuda.empty_cache()

        # Log losses
        train_loss = accum_train_loss / num_train_batches
        valid_loss = accum_valid_loss / num_valid_batches
        history_logger.log(epoch, train_loss, valid_loss)

        # Save debug outputs
        if epoch % save_debug_interval == 0:
            mask_debug_path = os.path.join(output_dir, "mask_debug_{}.npz".format(epoch))
            np.savez_compressed(mask_debug_path,
                                train_masks=train_masks_save,
                                train_idxs=train_idxs_save,
                                valid_masks=valid_masks_save,
                                valid_idxs=valid_idxs_save)

        if multi_gpu:
            separator_state_dict = separator.module.state_dict()
        else:
            separator_state_dict = separator.state_dict()

        # PyTorch saving recommendations: https://stackoverflow.com/a/49078976
        # Checkpoint every N epochs

        if epoch % checkpoint_interval == 0:
            separator_ckpt_path = os.path.join(output_dir, "separator_epoch-{}.pt".format(epoch))
            torch.save(separator_state_dict, separator_ckpt_path)

        # Save best model (w.r.t. validation loss)
        if history_logger.valid_loss_improved():
            # Checkpoint best model
            torch.save(separator_state_dict, separator_best_ckpt_path)
            early_stopping_wait = 0
        else:
            early_stopping_wait += 1

        # Always save latest states
        torch.save(separator_state_dict, separator_latest_ckpt_path)
        torch.save(optimizer.state_dict(), optimizer_latest_ckpt_path)

        # Early stopping
        if not early_stopping_flag and early_stopping_wait >= patience:
            shutil.copy(separator_best_ckpt_path, separator_earlystopping_ckpt_path)
            # Make sure that we only hit early stopping once
            early_stopping_flag = True

            # Terminate training if enabled
            if early_stopping_terminate:
                break

    # Save debug outputs for last epoch if they weren't already
    if epoch % save_debug_interval != 0:
        mask_debug_path = os.path.join(output_dir, "mask_debug_{}.npz".format(epoch))
        np.savez_compressed(mask_debug_path,
                            train_masks=train_masks_save,
                            train_idxs=train_idxs_save,
                            valid_masks=valid_masks_save,
                            valid_idxs=valid_idxs_save)

    # If early stopping was never hit, use best overall model
    if not early_stopping_flag:
        shutil.copy(separator_best_ckpt_path, separator_earlystopping_ckpt_path)

    print("Finished training. Results available at {}".format(output_dir))


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])

    # Load training config
    with open(args.train_config_path, 'r') as f:
        train_config = json.load(f)

    train(root_data_dir=args.root_data_dir,
          train_config=train_config,
          output_dir=args.output_dir,
          num_data_workers=args.num_data_workers,
          checkpoint_interval=args.checkpoint_interval,
          num_debug_examples=args.num_debug_examples,
          save_debug_interval=args.save_debug_interval,
          random_state=args.random_state,
          verbose=args.verbose)
