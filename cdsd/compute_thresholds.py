import argparse
import os
import json
import sys
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from .data import get_data_transforms, get_batch_input_key, CDSDDataset
from .models import construct_classifier
from sklearn.metrics import f1_score


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

    return parser.parse_args(args)


def estimate_class_weights(root_data_dir, train_config, output_dir,
                           num_data_workers=1):

    # Create output directory
    os.makedirs(output_dir)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump({"root_data_dir": root_data_dir,
                   "output_dir": output_dir,
                   "num_data_workers": num_data_workers,
                   "train_config": train_config}, f)

    input_transform = get_data_transforms(train_config)

    valid_dataset = CDSDDataset(root_data_dir,
                                subset='validate',
                                transform=input_transform)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=train_config["training"]["batch_size"],
                                  shuffle=True, pin_memory=True,
                                  num_workers=num_data_workers)
    num_train_batches = len(valid_dataloader)
    classifier = construct_classifier(train_config)

    input_key = get_batch_input_key(train_config)

    class_labels = valid_dataset.labels
    num_labels = len(valid_dataset.labels)

    y_list = []
    pred_list = []

    for batch in tqdm(valid_dataloader, total=num_train_batches):
        x = batch[input_key]
        y_list.append(batch["labels"].numpy())
        pred_list.append(classifier(x).numpy())

    y = np.vstack(y_list)
    pred = np.vstack(pred_list)

    threshold_dict = {}

    for class_idx in range(num_labels):
        y_cls = y[:, class_idx]
        pred_cls = pred[:, class_idx]

        threshold_candidates = np.sort([0.0] + list(pred_cls) + [1.0])
        f1_score_list = []
        for threshold in threshold_candidates:
            pred_cls_binary = (pred_cls >= threshold).astype(float)
            f1_score_list.append(f1_score(y_cls, pred_cls_binary))

        best_threshold = threshold_candidates[np.argmax(f1_score_list)]
        threshold_dict[class_labels[class_idx]] = best_threshold

    thresholds_path = os.path.join(output_dir, "class_detection_thresholds.json")

    with open(thresholds_path, 'w') as f:
        json.dump(thresholds_path, f)

    print("Finished computing thresholds. Available at {}".format(thresholds_path))


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])

    # Load training config
    with open(args.train_config_path, 'r') as f:
        train_config = json.load(f)

    with open(args.class_thresholds_path, 'r') as f:
        class_thresholds = json.load(f)

    estimate_class_weights(root_data_dir=args.root_data_dir,
                           train_config=train_config,
                           output_dir=args.output_dir,
                           num_data_workers=args.num_data_workers)
