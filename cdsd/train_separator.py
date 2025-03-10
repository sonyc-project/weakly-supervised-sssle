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
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from itertools import chain
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import get_data_transforms, CDSDDataset
from models import construct_separator, construct_classifier
from losses import get_mixture_loss_function
from utils import get_optimizer
from logs import CDSDHistoryLogger


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
    label_mode = train_config["training"]["label_mode"]
    assert label_mode in ("clip", "frame")

    train_dataset = CDSDDataset(root_data_dir,
                                subset='train',
                                transform=input_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, pin_memory=True,
                                  num_workers=num_data_workers)
    num_train_batches = len(train_dataloader)

    valid_dataset = CDSDDataset(root_data_dir,
                                subset='valid',
                                transform=input_transform)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                                  shuffle=True, pin_memory=True,
                                  num_workers=num_data_workers)
    num_valid_batches = len(valid_dataloader)

    # Set up models
    separator = construct_separator(train_config, dataset=train_dataset,
                                    device=device)
    classifier = construct_classifier(train_config, dataset=train_dataset,
                                      label_mode=label_mode, device=device)
    multi_gpu = False
    if torch.cuda.device_count() > 1:
        num_devices = torch.cuda.device_count()
        multi_gpu = True
        print("Using {} GPUs for training.".format(num_devices))
        separator = nn.DataParallel(separator)
        classifier = nn.DataParallel(classifier)

    separator.to(device)
    classifier.to(device)

    # Set up label downsampling
    input_num_frames = train_dataset.get_num_frames()
    output_num_frames = classifier.get_num_frames(input_num_frames)
    assert input_num_frames >= output_num_frames
    if label_mode == "frame" and input_num_frames > output_num_frames:
        # Set up max pool
        pool_size = input_num_frames // output_num_frames
        label_maxpool = nn.MaxPool1d(pool_size)
        if multi_gpu:
            label_maxpool = nn.DataParallel(label_maxpool)
        label_maxpool.to(device)
    else:
        label_maxpool = None

    class_prior_weighting = train_config["training"].get("class_prior_weighting", False)
    patience = train_config["training"].get("early_stopping_patience", 5)
    early_stopping_terminate = train_config["training"].get("early_stopping_terminate", False)
    separate_background = train_config["training"].get("separate_background", False)
    residual_background = train_config["training"].get("residual_background", False)
    classify_background = train_config["training"].get("classify_background", False)

    assert not (separate_background and residual_background)
    if classify_background and not (separate_background or residual_background):
        raise ValueError("Must separate background to classify it.")

    # JTC: Should we still provide params with requires_grad=False here?
    all_params = chain(separator.parameters(), classifier.parameters())
    trainable_params = (param for param in all_params if param.requires_grad)
    optimizer = get_optimizer(trainable_params, train_config)

    # Set up loss functions
    mixture_loss_fn = get_mixture_loss_function(train_config)
    # JTC: Look into BCEWithLogitsLoss, but for now just use BCELoss
    cls_loss_weight = train_config["losses"]["classification"]["weight"]


    # Set up history logging
    history_path = os.path.join(output_dir, "history.csv")
    history_logger = CDSDHistoryLogger(history_path)

    # Set up checkpoint paths
    separator_best_ckpt_path = os.path.join(output_dir, "separator_best.pt")
    classifier_best_ckpt_path = os.path.join(output_dir, "classifier_best.pt")
    separator_latest_ckpt_path = os.path.join(output_dir, "separator_latest.pt")
    classifier_latest_ckpt_path = os.path.join(output_dir, "classifier_latest.pt")
    separator_earlystopping_ckpt_path = os.path.join(output_dir, "separator_earlystopping.pt")
    classifier_earlystopping_ckpt_path = os.path.join(output_dir, "classifier_earlystopping.pt")
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
        save_config["classifier"]["best_path"] = classifier_best_ckpt_path
        save_config["classifier"]["latest_path"] = classifier_latest_ckpt_path
        save_config["classifier"]["earlystopping_path"] = classifier_earlystopping_ckpt_path
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
        accum_train_mix_loss = 0.0
        accum_train_cls_loss = 0.0
        accum_train_total_loss = 0.0
        accum_valid_mix_loss = 0.0
        accum_valid_cls_loss = 0.0
        accum_valid_total_loss = 0.0

        train_masks_save = None
        train_idxs_save = None
        valid_masks_save = None
        valid_idxs_save = None

        print(" **** Training ****")
        # Set models to train mode
        separator.train()
        classifier.train()
        torch.cuda.empty_cache()
        for batch_idx, batch in enumerate(tqdm(train_dataloader, total=num_train_batches, disable=(not verbose))):
            x = batch["audio_data"].to(device)
            if label_mode == "frame":
                cls_target_labels_raw = batch["frame_labels"].to(device)
                cls_target_labels = cls_target_labels_raw
                # Downsample labels if necessary
                if label_maxpool is not None:
                    cls_target_labels = label_maxpool(cls_target_labels.transpose(1, 2)).transpose(1, 2)

                class_weights = torch.zeros_like(cls_target_labels, device=device)
                frame_labels = None
                for class_idx in range(train_dataset.num_labels):
                    frame_labels = cls_target_labels[..., class_idx]
                    p = train_dataset.class_frame_priors[class_idx]
                    class_weights[..., class_idx][frame_labels.bool()] = 1.0 / p
                    class_weights[..., class_idx][(-frame_labels + 1).bool()] = 1.0 / (1 - p)
                del frame_labels
            else:
                cls_target_labels_raw = batch["clip_labels"].to(device)
                cls_target_labels = cls_target_labels_raw
                class_weights = None

            energy_mask = batch["energy_mask"].to(device)

            if epoch == 0 and batch_idx == 0:
                print("Input size: {}".format(x.size()))

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass through separator
            masks = separator(x)

            mixture_labels = cls_target_labels_raw
            if separate_background:
                # Add "background labels" if we're also explicitly separating
                # the background
                mixture_labels = torch.cat((mixture_labels, torch.ones(cls_target_labels_raw.shape[:-1] + (1,))), dim=-1)

            # Compute mixture loss for separator
            train_mix_loss = mixture_loss_fn(x,
                                             labels=mixture_labels,
                                             masks=masks,
                                             energy_mask=energy_mask)

            # Pass mixture through classifier
            mix_cls_output = classifier(x)

            # Compute mixture classification_loss
            if class_prior_weighting and label_mode == 'frame':
                cls_bce = F.binary_cross_entropy(mix_cls_output,
                                                        cls_target_labels,
                                                        reduction='none')
                cls_bce = cls_bce * class_weights
                train_cls_loss = cls_bce.mean()
            else:
                cls_bce = None
                train_cls_loss = F.binary_cross_entropy(mix_cls_output,
                                                        cls_target_labels)

            # Pass reconstructed sources through classifier
            for idx in range(train_dataset.num_labels):
                mask = masks[..., idx]
                x_masked = (x * mask).to(device)
                src_cls_output = classifier(x_masked)

                # Create target
                target = torch.zeros_like(cls_target_labels).to(device)
                target[..., idx] = cls_target_labels[..., idx].to(device)

                # Accumulate classification loss for each source type
                if class_prior_weighting and label_mode == 'frame':
                    cls_bce = F.binary_cross_entropy(src_cls_output,
                                                     target,
                                                     reduction='none')
                    cls_bce = cls_bce * class_weights
                    train_cls_loss = train_cls_loss + cls_bce.mean()
                else:
                    cls_bce = None
                    train_cls_loss = train_cls_loss + F.binary_cross_entropy(src_cls_output, target)

            if classify_background:
                if separate_background:
                    mask = masks[..., -1]
                elif residual_background:
                    mask = torch.relu(-masks.sum(dim=-1) + 1.0)
                else:
                    raise ValueError('Classifying background with no method for obtaining background.')
                x_masked = (x * mask).to(device)
                src_cls_output = classifier(x_masked)

                # Background should be all zeros for all classes
                target = torch.zeros_like(cls_target_labels).to(device)
                # Accumulate classification loss for each source type
                if class_prior_weighting and label_mode == 'frame':
                    cls_bce = F.binary_cross_entropy(src_cls_output,
                                                     target,
                                                     reduction='none')
                    cls_bce = cls_bce * class_weights
                    train_cls_loss = train_cls_loss + cls_bce.mean()
                else:
                    cls_bce = None
                    train_cls_loss = train_cls_loss + F.binary_cross_entropy(src_cls_output, target)

            # Accumulate loss
            train_total_loss = train_mix_loss + train_cls_loss * cls_loss_weight

            # Backprop
            train_total_loss.backward()
            optimizer.step()

            # Accumulate loss for epoch
            accum_train_mix_loss += train_mix_loss.item()
            accum_train_cls_loss += train_cls_loss.item()
            accum_train_total_loss += train_total_loss.item()

            # Save debug outputs (for first N training examples)
            if epoch % save_debug_interval == 0 and batch_idx == 0:
                train_masks_save = masks[:num_debug_examples].cpu().detach().numpy()
                train_idxs_save = batch['index'][:num_debug_examples].cpu().numpy()

            # Cleanup
            del x, cls_bce, cls_target_labels, mixture_labels, \
                cls_target_labels_raw, masks, energy_mask, batch, mask, \
                x_masked, mix_cls_output, src_cls_output, train_cls_loss, \
                train_mix_loss, train_total_loss, class_weights
            torch.cuda.empty_cache()

        # Evaluate on validation set
        print(" **** Validation ****")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(valid_dataloader, total=num_valid_batches, disable=(not verbose))):
                x = batch["audio_data"].to(device)
                if label_mode == "frame":
                    cls_target_labels_raw = batch["frame_labels"].to(device)
                    cls_target_labels = cls_target_labels_raw
                    # Downsample labels if necessary
                    if label_maxpool is not None:
                        cls_target_labels = label_maxpool(cls_target_labels.transpose(1, 2)).transpose(1, 2)

                    class_weights = torch.zeros_like(cls_target_labels, device=device)
                    frame_labels = None
                    for class_idx in range(train_dataset.num_labels):
                        frame_labels = cls_target_labels[..., class_idx]
                        p = train_dataset.class_frame_priors[class_idx]
                        class_weights[..., class_idx][frame_labels.bool()] = 1.0 / p
                        class_weights[..., class_idx][(-frame_labels + 1).bool()] = 1.0 / (1 - p)
                    del frame_labels
                else:
                    cls_target_labels_raw = batch["clip_labels"].to(device)
                    cls_target_labels = cls_target_labels_raw
                    class_weights = None

                energy_mask = batch["energy_mask"].to(device)

                # Set models to eval mode
                separator.eval()
                classifier.eval()

                # Forward pass through separator
                masks = separator(x)

                mixture_labels = cls_target_labels_raw
                if separate_background:
                    # Add "background labels" if we're also explicitly separating
                    # the background
                    mixture_labels = torch.cat((mixture_labels, torch.ones(cls_target_labels_raw.shape[:-1] + (1,))), dim=-1)

                # Compute mixture loss for separator
                valid_mix_loss = mixture_loss_fn(x,
                                                 labels=mixture_labels,
                                                 masks=masks,
                                                 energy_mask=energy_mask)

                # Pass mixture through classifier
                mix_cls_output = classifier(x)

                # Compute mixture classification_loss
                if class_prior_weighting and label_mode == 'frame':
                    cls_bce = F.binary_cross_entropy(mix_cls_output,
                                                     cls_target_labels,
                                                     reduction='none')
                    cls_bce = cls_bce * class_weights
                    valid_cls_loss = cls_bce.mean()
                else:
                    cls_bce = None
                    valid_cls_loss = F.binary_cross_entropy(mix_cls_output,
                                                            cls_target_labels)

                # Pass reconstructed sources through classifier
                for idx in range(train_dataset.num_labels):
                    mask = masks[..., idx]
                    x_masked = (x * mask).to(device)
                    src_cls_output = classifier(x_masked)

                    # Create target
                    target = torch.zeros_like(cls_target_labels).to(device)
                    target[..., idx] = cls_target_labels[..., idx].to(device)

                    # Accumulate classification loss for each source type
                    if class_prior_weighting and label_mode == 'frame':
                        cls_bce = F.binary_cross_entropy(src_cls_output,
                                                         target,
                                                         reduction='none')
                        cls_bce = cls_bce * class_weights
                        valid_cls_loss = valid_cls_loss + cls_bce.mean()
                    else:
                        cls_bce = None
                        valid_cls_loss = valid_cls_loss + F.binary_cross_entropy(src_cls_output, target)

                if classify_background:
                    if separate_background:
                        mask = masks[..., -1]
                    elif residual_background:
                        mask = torch.relu(-masks.sum(dim=-1) + 1.0)
                    else:
                        raise ValueError('Classifying background with no method for obtaining background.')
                    x_masked = (x * mask).to(device)
                    src_cls_output = classifier(x_masked)

                    # Background should be all zeros for all classes
                    target = torch.zeros_like(cls_target_labels).to(device)
                    # Accumulate classification loss for each source type
                    if class_prior_weighting and label_mode == 'frame':
                        cls_bce = F.binary_cross_entropy(src_cls_output,
                                                         target,
                                                         reduction='none')
                        cls_bce = cls_bce * class_weights
                        valid_cls_loss = valid_cls_loss + cls_bce.mean()
                    else:
                        cls_bce = None
                        valid_cls_loss = valid_cls_loss + F.binary_cross_entropy(src_cls_output, target)

                # Accumulate loss
                valid_total_loss = valid_mix_loss + valid_cls_loss * cls_loss_weight

                # Accumulate loss for epoch
                accum_valid_mix_loss += valid_mix_loss.item()
                accum_valid_cls_loss += valid_cls_loss.item()
                accum_valid_total_loss += valid_total_loss.item()

                # Save debug outputs (for first N training examples)
                if epoch % save_debug_interval == 0 and batch_idx == 0:
                    valid_masks_save = masks[:num_debug_examples].cpu().detach().numpy()
                    valid_idxs_save = batch['index'][:num_debug_examples].cpu().numpy()

                # Help garbage collection
                del x, cls_bce, cls_target_labels, energy_mask, masks, mixture_labels, \
                    batch, mask, x_masked, mix_cls_output, src_cls_output, \
                    valid_cls_loss, valid_mix_loss, valid_total_loss, class_weights
                torch.cuda.empty_cache()

        # Log losses
        train_mix_loss = accum_train_mix_loss / num_train_batches
        train_cls_loss = accum_train_cls_loss / num_train_batches
        train_tot_loss = accum_train_total_loss / num_train_batches
        valid_mix_loss = accum_valid_mix_loss / num_valid_batches
        valid_cls_loss = accum_valid_cls_loss / num_valid_batches
        valid_tot_loss = accum_valid_total_loss / num_valid_batches
        history_logger.log(epoch, train_mix_loss, train_cls_loss, train_tot_loss,
                           valid_mix_loss, valid_cls_loss, valid_tot_loss)

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
            classifier_state_dict = classifier.module.state_dict()
        else:
            separator_state_dict = separator.state_dict()
            classifier_state_dict = classifier.state_dict()

        # PyTorch saving recommendations: https://stackoverflow.com/a/49078976
        # Checkpoint every N epochs

        if epoch % checkpoint_interval == 0:
            separator_ckpt_path = os.path.join(output_dir, "separator_epoch-{}.pt".format(epoch))
            classifier_ckpt_path = os.path.join(output_dir, "classifier_epoch-{}.pt".format(epoch))
            torch.save(separator_state_dict, separator_ckpt_path)
            torch.save(classifier_state_dict, classifier_ckpt_path)

        # Save best model (w.r.t. validation loss)
        if history_logger.valid_loss_improved():
            # Checkpoint best model
            torch.save(separator_state_dict, separator_best_ckpt_path)
            torch.save(classifier_state_dict, classifier_best_ckpt_path)
            early_stopping_wait = 0
        else:
            early_stopping_wait += 1

        # Always save latest states
        torch.save(separator_state_dict, separator_latest_ckpt_path)
        torch.save(classifier_state_dict, classifier_latest_ckpt_path)
        torch.save(optimizer.state_dict(), optimizer_latest_ckpt_path)

        # Early stopping
        if not early_stopping_flag and early_stopping_wait >= patience:
            shutil.copy(separator_best_ckpt_path, separator_earlystopping_ckpt_path)
            shutil.copy(classifier_best_ckpt_path, classifier_earlystopping_ckpt_path)
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
        shutil.copy(classifier_best_ckpt_path, classifier_earlystopping_ckpt_path)

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
