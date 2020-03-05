import os
import sys
import pandas as pd
import subprocess
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from tqdm import tqdm


US8K_LABELS = [
    'air_conditioner',
    'car_horn',
    'children_playing',
    'dog_bark',
    'drilling',
    'engine_idling',
    'gun_shot',
    'jackhammer',
    'siren',
    'street_music'
]

# None indicates ignore
US8K_TO_SONYCUST_MAP = {
    'air_conditioner': None,
    'car_horn': 'alert-signal',
    'children_playing': 'human-voice',
    'dog_bark': 'dog',
    'drilling': None,
    'engine_idling': 'engine',
    'gun_shot': None,
    'jackhammer': 'machinery-impact',
    'siren': 'alert-signal',
    'street_music': 'music'
}

TRAIN_FOLDS = {1, 2, 3, 4, 5, 6}
VALID_FOLDS = {7, 8}
TEST_FOLDS = {9, 10}


def parse_arguments(args):
    parser = ArgumentParser(sys.argv[0],
                            description="Preprocess UrbanSound8K to be in a format usable by Scaper to synthesize soundscapes.",
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('us8k_folder', type=str,
                        help='Path to UrbanSound8K dataset.')
    parser.add_argument('out_folder', type=str,
                        help='Output directory where audio files will be saved.')
    parser.add_argument('--ust-mapping', action='store_true',
                        help='If true, uses SONYC-UST mappings.')

    return parser.parse_args(args)


def run(us8k_dir, out_dir, ust_mapping=False):
    metadata_path = os.path.join(us8k_dir, 'metadata', 'UrbanSound8K.csv')
    df = pd.read_csv(metadata_path)

    train_dir = os.path.join(out_dir, 'train')
    valid_dir = os.path.join(out_dir, 'valid')
    test_dir = os.path.join(out_dir, 'test')

    if ust_mapping:
        # Create all of the output directories
        for subset_dir in (train_dir, valid_dir, test_dir):
            for label in US8K_TO_SONYCUST_MAP.values():
                if label is not None:
                    # Only create label directories for the labels that will be
                    # used
                    label_dir = os.path.join(subset_dir, label)
                    os.makedirs(label_dir, exist_ok=True)

    num_files = len(df)
    for _, row in tqdm(df.iterrows(), total=num_files):
        # Ignore "background" events
        if row['salience'] == 2:
            continue

        # Determine which subset this file belongs to from the fold
        fold = row['fold']
        if fold in TRAIN_FOLDS:
            subset_dir = train_dir
        elif fold in VALID_FOLDS:
            subset_dir = valid_dir
        elif fold in TEST_FOLDS:
            subset_dir = test_dir
        else:
            raise ValueError('Invalid fold: {}'.format(fold))

        # Map US8K label to SONYC-UST label
        us8k_label = US8K_LABELS[row['classID']]

        if ust_mapping:
            label = US8K_TO_SONYCUST_MAP[us8k_label]
        else:
            label = us8k_label

        # Ignore "None" labels
        if label is None:
            continue
        fname = row['slice_file_name']
        src_path = os.path.join(us8k_dir, 'audio', 'fold{}'.format(fold), fname)
        dst_path = os.path.join(subset_dir, label, fname)

        if not ust_mapping:
            # Make sure output dir exists
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # Process the file with FFMPEG to ensure that the formats are uniform
        cmd_args = ["ffmpeg", "-i", src_path, dst_path]
        res = subprocess.run(cmd_args, capture_output=True)
        if res.returncode != 0:
            err_msg = "Error processing {}:\n{}\n{}"
            raise OSError(err_msg.format(res.stdout, res.stderr))


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    run(us8k_dir=args.us8k_folder,
        out_dir=args.out_folder,
        ust_mapping=args.ust_mapping,
        use_symlinks=args.use_symlinks)

