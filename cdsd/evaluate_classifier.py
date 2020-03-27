import argparse
import json
import os
import sys
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import get_data_transforms, CDSDDataset
from models import construct_separator, construct_classifier


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
    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Memory hack: https://discuss.pytorch.org/t/solved-pytorch-conv2d-consumes-more-gpu-memory-than-tensorflow/28998/2
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

    # Create output directory
    output_dir = output_dir or train_config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Set up data loaders
    input_transform = get_data_transforms(train_config)
    train_dataset = CDSDDataset(root_data_dir,
                                subset='train',
                                transform=input_transform)

    # Set up models
    classifier = construct_classifier(train_config,
                                      dataset=train_dataset,
                                      require_init=True,
                                      trainable=False)
    if torch.cuda.device_count() > 1:
        print("Using {} GPUs for evaluation.".format(torch.cuda.device_count()))
        classifier = nn.DataParallel(classifier)
    classifier.to(device)
    classifier.eval()

    with torch.no_grad():
        for subset in ('train', 'valid', 'test'):
            print('====== Evaluating subset "{}" ======'.format(subset))
            dataset = CDSDDataset(root_data_dir,
                                  subset=subset,
                                  transform=input_transform)
            dataloader = DataLoader(dataset, batch_size=train_config["training"]["batch_size"],
                                    shuffle=False, pin_memory=True,
                                    num_workers=num_data_workers)
            num_batches = len(dataloader)

            # Initialize results lists
            subset_results = {label + "_presence_gt": [] for label in dataset.labels}
            subset_results.update({label + "_presence_pred": [] for label in dataset.labels})
            subset_results["filenames"] = list(dataset.files) # Assuming that dataloader preserves order

            subset_results_path = os.path.join(output_dir, "classification_results_{}.csv".format(subset))

            for batch in tqdm(dataloader, total=num_batches):
                x = batch["audio_data"].to(device)
                labels = batch["labels"].to(device)
                # Run classifier on mixture for later analysis
                pred = classifier(x)

                for label_idx, label in enumerate(train_dataset.labels):
                    subset_results[label + "_presence_gt"] += labels[:, label_idx].tolist()
                    subset_results[label + "_presence_pred"] += pred[:, label_idx].tolist()

                del x, labels, pred, batch

            # Save results as CSV
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
