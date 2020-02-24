import argparse
import os
import json
import sys
import oyaml as yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .data import get_data_transforms, get_batch_input_key, CDSDDataset
from .models import construct_classifier


def parse_arguments(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('root_data_dir',
                        type=str,
                        help='Path to dataset directory')

    parser.add_argument('train-config_path',
                        type=str,
                        help='Path to training configuration JSON file')

    parser.add_argument('class_thresholds_path',
                        type=str,
                        help='Path to class-wise detection thresholds')

    parser.add_argument('ust_taxonomy_path', type=str,
                        help='Path to SONYC-UST taxonomy file.')

    parser.add_argument('output_dir',
                        type=str,
                        help='Path where outputs will be saved')

    parser.add_argument('-n', '--num-data-workers',
                        type=int, default=1,
                        help='Number of workers used for data loading.')

    return parser.parse_args(args)


def estimate_class_weights(root_data_dir, train_config, class_thresholds, taxonomy,
                           output_dir, num_data_workers=1):

    # Create output directory
    os.makedirs(output_dir)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump({"root_data_dir": root_data_dir,
                   "output_dir": output_dir,
                   "num_data_workers": num_data_workers,
                   "train_config": train_config}, f)

    input_transform = get_data_transforms(train_config)

    train_dataset = CDSDDataset(root_data_dir,
                                subset='train',
                                transform=input_transform)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=train_config["training"]["batch_size"],
                                  shuffle=True, pin_memory=True,
                                  num_workers=num_data_workers)
    num_train_batches = len(train_dataloader)
    classifier = construct_classifier(train_config)

    input_key = get_batch_input_key(train_config)

    accum_clip_presence = None
    accum_frame_presence = None
    accum_thresholded_frame_presence = None
    total_frames = 0
    total_clips = 0

    num_labels = train_dataset.num_labels
    class_labels = train_dataset.labels

    # Convert class threshold dictionary to torch tensor
    class_thresholds = torch.FloatTensor([class_thresholds[label] for label in class_labels]).view(1, 1, -1)

    for batch in tqdm(train_dataloader, total=num_train_batches):
        x = batch[input_key]
        labels = batch["labels"]
        batch_size = x.size()[0]
        num_frames = x.size()[-1]

        # Use labels to reduce false positives
        output = classifier(x) * labels.view(batch_size, 1, num_labels)

        # Use thresholds to binarize estimates of class presence
        thresholded_output = (output >= class_thresholds).float()

        clip_presence = labels.sum(dim=0).numpy()
        frame_presence = output.view(batch_size * num_frames, num_labels).sum(dim=0).numpy()
        thresholded_frame_presence = thresholded_output.view(batch_size * num_frames, num_labels).sum(dim=0).numpy()

        if accum_clip_presence is not None:
            accum_clip_presence += clip_presence
            accum_frame_presence += frame_presence
            accum_thresholded_frame_presence += thresholded_frame_presence
        else:
            accum_clip_presence = clip_presence
            accum_frame_presence = frame_presence
            accum_thresholded_frame_presence = thresholded_frame_presence

        total_clips += batch_size
        total_frames += batch_size * num_frames

    clip_class_weights = accum_clip_presence / total_clips
    frame_class_weights = accum_frame_presence / total_frames
    thresholded_frame_class_weights = accum_thresholded_frame_presence / total_frames

    # Save class weights
    clip_class_weight_dict = {label: weight for label, weight in zip(class_labels, clip_class_weights)}
    frame_class_weight_dict = {label: weight for label, weight in zip(class_labels, frame_class_weights)}
    thresholded_frame_class_weight_dict = {label: weight for label, weight in zip(class_labels, thresholded_frame_class_weights)}
    clip_output_path = os.path.join(output_dir, "class_weights_clip.json")
    frame_output_path = os.path.join(output_dir, "class_weights_frame.json")
    thresholded_frame_output_path = os.path.join(output_dir, "class_weights_thresholded-frame.json")

    with open(clip_output_path, 'w') as f:
        json.dump(clip_class_weight_dict, f)
    with open(frame_output_path, 'w') as f:
        json.dump(frame_class_weight_dict, f)
    with open(thresholded_frame_output_path, 'w') as f:
        json.dump(thresholded_frame_class_weight_dict, f)

    print("Finished training. Results available at {}".format(output_dir))


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])

    # Load training config
    with open(args.train_config_path, 'r') as f:
        train_config = json.load(f)

    with open(args.class_thresholds_path, 'r') as f:
        class_thresholds = json.load(f)

    with open(args.ust_taxonomy_path, 'r') as f:
        taxonomy = yaml.load(f)

    estimate_class_weights(root_data_dir=args.root_data_dir,
                           train_config=train_config,
                           taxonomy=taxonomy,
                           output_dir=args.output_dir,
                           class_thresholds=class_thresholds,
                           num_data_workers=args.num_data_workers)
