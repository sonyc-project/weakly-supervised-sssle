import argparse
import os
import json
import sys
import torch
import torch.nn as nn
from copy import deepcopy
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import get_data_transforms, CDSDDataset
from models import construct_classifier
from utils import get_optimizer
from logs import ClassifierHistoryLogger


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
    classifier = construct_classifier(train_config, dataset=train_dataset)
    multi_gpu = False
    if torch.cuda.device_count() > 1:
        print("Using {} GPUs for training.".format(torch.cuda.device_count()))
        multi_gpu = True
        classifier = nn.DataParallel(classifier)
    classifier.to(device)

    # JTC: Should we still provide params with requires_grad=False here?
    optimizer = get_optimizer(classifier.parameters(), train_config)

    # Set up loss functions
    # JTC: Look into BCEWithLogitsLoss, but for now just use BCELoss
    bce_loss_obj = nn.BCELoss()

    # Set up history logging
    history_path = os.path.join(output_dir, "history.csv")
    history_logger = ClassifierHistoryLogger(history_path)

    # Set up checkpoint paths
    classifier_best_ckpt_path = os.path.join(output_dir, "classifier_best.pt")
    classifier_latest_ckpt_path = os.path.join(output_dir, "classifier_latest.pt")
    optimizer_latest_ckpt_path = os.path.join(output_dir, "optimizer_latest.pt")

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        save_config = deepcopy(train_config)
        save_config["root_data_dir"] = root_data_dir
        save_config["output_dir"] = output_dir
        save_config["num_data_workers"] = num_data_workers
        save_config["classifier"]["best_path"] = classifier_best_ckpt_path
        save_config["classifier"]["latest_path"] = classifier_latest_ckpt_path
        json.dump(save_config, f)

    num_epochs = train_config["training"]["num_epochs"]
    for epoch in range(num_epochs):
        print("=============== Epoch {}/{} =============== ".format(epoch+1, num_epochs))
        accum_train_loss = 0.0
        accum_valid_loss = 0.0

        print(" **** Training ****")
        for batch in tqdm(train_dataloader, total=num_train_batches):
            x = batch["audio_data"].to(device)
            labels = batch["labels"].to(device)
            output = classifier(x)

            train_loss = bce_loss_obj(output, labels)
            train_loss.backward()
            optimizer.step()

            # Accumulate loss for epoch
            accum_train_loss += train_loss.item()

        # Evaluate on validation set
        print(" **** Validation ****")
        for batch in tqdm(valid_dataloader, total=num_valid_batches):
            x = batch["audio_data"].to(device)
            labels = batch["labels"].to(device)
            output = classifier(x)

            valid_loss = bce_loss_obj(output, labels)

            # Accumulate loss for epoch
            accum_valid_loss += valid_loss.item()

        train_loss = accum_train_loss / num_train_batches
        valid_loss = accum_valid_loss / num_valid_batches
        history_logger.log(epoch, train_loss, valid_loss)

        # If using DataParallel, get model from inside wrapper
        if multi_gpu:
            classifier_state_dict = classifier.module.state_dict()
        else:
            classifier_state_dict = classifier.state_dict()

        # PyTorch saving recommendations: https://stackoverflow.com/a/49078976
        # Checkpoint every N epochs
        if epoch % checkpoint_interval:
            classifier_ckpt_path = os.path.join(output_dir, "classifier_epoch-{}.pt".format(epoch))
            torch.save(classifier_state_dict, classifier_ckpt_path)

        # Save best model (w.r.t. validation loss)
        if history_logger.valid_loss_improved():
            # Checkpoint best model
            torch.save(classifier_state_dict, classifier_best_ckpt_path)

        # Always save latest states
        torch.save(classifier_state_dict, classifier_latest_ckpt_path)
        torch.save(optimizer.state_dict(), optimizer_latest_ckpt_path)

    history_logger.close()
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
