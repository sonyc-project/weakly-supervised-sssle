import argparse
import os
import json
import sys
import torch
import torch.nn as nn
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

    return parser.parse_args(args)


def train(root_data_dir, train_config, output_dir, num_data_workers=1, checkpoint_interval=10):
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
    separator = construct_separator(train_config, dataset=train_dataset)
    classifier = construct_classifier(train_config, dataset=train_dataset)
    multi_gpu = False
    if torch.cuda.device_count() > 1:
        num_devices = torch.cuda.device_count()
        multi_gpu = True
        print("Using {} GPUs for training.".format(num_devices))
        separator = nn.DataParallel(separator)
        classifier = nn.DataParallel(classifier)

    separator.to(device)
    classifier.to(device)

    # JTC: Should we still provide params with requires_grad=False here?
    optimizer = get_optimizer(chain(separator.parameters(), classifier.parameters()),
                              train_config)

    # Set up loss functions
    mixture_loss_fn = get_mixture_loss_function(train_config)
    mixture_loss_weight = train_config["losses"]["mixture"]["weight"]
    # JTC: Look into BCEWithLogitsLoss, but for now just use BCELoss
    bce_loss_obj = nn.BCELoss()
    cls_loss_weight = train_config["losses"]["classification"]["weight"]

    # Set up history logging
    history_path = os.path.join(output_dir, "history.csv")
    history_logger = CDSDHistoryLogger(history_path)

    # Set up checkpoint paths
    separator_best_ckpt_path = os.path.join(output_dir, "separator_best.pt")
    classifier_best_ckpt_path = os.path.join(output_dir, "classifier_best.pt")
    separator_latest_ckpt_path = os.path.join(output_dir, "separator_latest.pt")
    classifier_latest_ckpt_path = os.path.join(output_dir, "classifier_latest.pt")
    optimizer_latest_ckpt_path = os.path.join(output_dir, "optimizer_latest.pt")

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        save_config = deepcopy(train_config)
        save_config["root_data_dir"] = root_data_dir
        save_config["output_dir"] = output_dir
        save_config["num_data_workers"] = num_data_workers
        save_config["separator"]["best_path"] = separator_best_ckpt_path
        save_config["separator"]["latest_path"] = separator_latest_ckpt_path
        save_config["classifier"]["best_path"] = classifier_best_ckpt_path
        save_config["classifier"]["latest_path"] = classifier_latest_ckpt_path
        json.dump(save_config, f)

    num_epochs = train_config["training"]["num_epochs"]
    for epoch in range(num_epochs):
        print("=============== Epoch {}/{} =============== ".format(epoch + 1, num_epochs))
        accum_train_mix_loss = 0.0
        accum_train_cls_loss = 0.0
        accum_train_total_loss = 0.0
        accum_valid_mix_loss = 0.0
        accum_valid_cls_loss = 0.0
        accum_valid_total_loss = 0.0

        print(" **** Training ****")
        # Set models to train mode
        separator.train()
        classifier.train()
        for batch_idx, batch in enumerate(tqdm(train_dataloader, total=num_train_batches)):
            x = batch["audio_data"].to(device)
            labels = batch["labels"].to(device)

            if epoch == 0 and batch_idx == 0:
                print("Input size: {}".format(x.size()))

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass through separator
            masks = separator(x)

            # Pass reconstructed sources through classifier
            train_cls_loss = None
            for idx in range(train_dataset.num_labels):
                mask = masks[..., idx]
                x_masked = (x * mask).to(device)
                output = classifier(x_masked)

                # Create target
                target = torch.zeros_like(labels).to(device)
                target[:, idx] = labels[:, idx].to(device)

                # Accumulate classification loss for each source type
                if train_cls_loss is None:
                    train_cls_loss = bce_loss_obj(output, target)
                else:
                    train_cls_loss += bce_loss_obj(output, target)

            assert train_cls_loss is not None

            train_mix_loss = mixture_loss_fn(x, labels, masks)

            # Accumulate loss
            train_total_loss = train_mix_loss * mixture_loss_weight + train_cls_loss * cls_loss_weight

            # Backprop
            train_total_loss.backward()
            optimizer.step()

            # Accumulate loss for epoch
            accum_train_mix_loss += train_mix_loss.item()
            accum_train_cls_loss += train_cls_loss.item()
            accum_train_total_loss += train_total_loss.item()

            # Cleanup
            del x, labels, masks, batch, mask, x_masked, output, \
                train_cls_loss, train_mix_loss, train_total_loss

        # Evaluate on validation set
        print(" **** Validation ****")
        with torch.no_grad():
            for batch in tqdm(valid_dataloader, total=num_valid_batches):
                x = batch["audio_data"].to(device)
                labels = batch["labels"].to(device)

                # Set models to eval mode
                separator.eval()
                classifier.eval()

                # Forward pass through separator
                masks = separator(x)

                # Pass reconstructed sources through classifier
                valid_cls_loss = None
                for idx in range(train_dataset.num_labels):
                    mask = masks[..., idx]
                    x_masked = (x * mask).to(device)
                    output = classifier(x_masked)

                    # Create target
                    target = torch.zeros_like(labels).to(device)
                    target[:, idx] = labels[:, idx].to(device)

                    # Accumulate classification loss for each source type
                    if valid_cls_loss is None:
                        valid_cls_loss = bce_loss_obj(output, target)
                    else:
                        valid_cls_loss += bce_loss_obj(output, target)

                assert valid_cls_loss is not None

                valid_mix_loss = mixture_loss_fn(x, labels, masks)
                valid_total_loss = valid_mix_loss * mixture_loss_weight + valid_cls_loss * cls_loss_weight

                # Accumulate loss for epoch
                accum_valid_mix_loss += valid_mix_loss.item()
                accum_valid_cls_loss += valid_cls_loss.item()
                accum_valid_total_loss += valid_total_loss.item()

                # Help garbage collection
                del batch

        train_mix_loss = accum_train_mix_loss / num_train_batches
        train_cls_loss = accum_train_cls_loss / num_train_batches
        train_tot_loss = accum_train_total_loss / num_train_batches
        valid_mix_loss = accum_valid_mix_loss / num_valid_batches
        valid_cls_loss = accum_valid_cls_loss / num_valid_batches
        valid_tot_loss = accum_valid_total_loss / num_valid_batches
        history_logger.log(epoch, train_mix_loss, train_cls_loss, train_tot_loss,
                           valid_mix_loss, valid_cls_loss, valid_tot_loss)

        if multi_gpu:
            separator_state_dict = separator.module.state_dict()
            classifier_state_dict = classifier.module.state_dict()
        else:
            separator_state_dict = separator.state_dict()
            classifier_state_dict = classifier.state_dict()

        # PyTorch saving recommendations: https://stackoverflow.com/a/49078976
        # Checkpoint every N epochs
        h
        if epoch % checkpoint_interval:
            separator_ckpt_path = os.path.join(output_dir, "separator_epoch-{}.pt".format(epoch))
            classifier_ckpt_path = os.path.join(output_dir, "classifier_epoch-{}.pt".format(epoch))
            torch.save(separator_state_dict, separator_ckpt_path)
            torch.save(classifier_state_dict, classifier_ckpt_path)

        # Save best model (w.r.t. validation loss)
        if history_logger.valid_loss_improved():
            # Checkpoint best model
            torch.save(separator_state_dict, separator_best_ckpt_path)
            torch.save(classifier_state_dict, classifier_best_ckpt_path)

        # Always save latest states
        torch.save(separator_state_dict, separator_latest_ckpt_path)
        torch.save(classifier_state_dict, classifier_latest_ckpt_path)
        torch.save(optimizer.state_dict(), optimizer_latest_ckpt_path)

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
          checkpoint_interval=args.checkpoint_interval)
