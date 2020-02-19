import os
import sys
import pandas as pd
import shutil
import soundfile as sf
import oyaml as yaml
import jams
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from tqdm import tqdm
from .preprocess_us8k import US8K_TO_SONYCUST_MAP


LABELS = sorted([x for x in US8K_TO_SONYCUST_MAP.values() if x is not None])


def parse_arguments(args):
    parser = ArgumentParser(sys.argv[0],
                            description="Preprocess SONYC-UST mixtures to be in same format as synthesized mixtures.",
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('ust_annotation_path', type=str,
                        help='Path to SONYC-UST annotation file.')
    parser.add_argument('ust_taxonomy_path', type=str,
                        help='Path to SONYC-UST annotation file.')
    parser.add_argument('ust_folder', type=str,
                        help='Path to SONYC-UST dataset.')
    parser.add_argument('out_folder', type=str,
                        help='Output directory where audio files will be saved.')
    parser.add_argument('--use-symlinks', action='store_true',
                        help='If true, create symlinks to audio files to avoid copying audio.')

    return parser.parse_args(args)


def run(annotation_path, taxonomy_path, data_dir, out_dir, use_symlinks=False):
    ann_df = pd.read_csv(annotation_path)
    # Restrict to verified annotations
    ann_df = ann_df[ann_df["annotator_id"] == 0].sort_values('audio_filename')
    ann_df = ann_df[['split', 'audio_filename']].drop_duplicates()

    os.makedirs(out_dir, exist_ok=True)

    # Load taxonomy
    with open(taxonomy_path, 'r') as f:
        taxonomy = yaml.load(f)
    label_to_id = {v: k for k, v in taxonomy['coarse']}

    for _, row in tqdm(ann_df.iterrows()):
        filename = row['audio_filename']
        split_str = row['split']

        src_audio_path = os.path.join(data_dir, split_str, filename)
        dst_audio_path = os.path.join(out_dir, filename)
        dst_jams_path = os.path.join(out_dir, os.path.splitext(filename)[0] + '.jams')

        # Copy audio
        if not use_symlinks:
            shutil.copy2(src_audio_path, dst_audio_path)
        else:
            os.symlink(src_audio_path, dst_audio_path)

        # Load audio to get duration
        audio, sr = sf.read(src_audio_path)
        duration = audio.shape[-1] / sr

        # Create JAMS file to be compatible with synthesized mixtures
        # Since we don't have strong event annotations, we are just assuming
        # all events take place throughout the duration of the clips
        jam = jams.JAMS()
        jam.file_metadata.duration = duration
        ann = jams.Annotation(namespace='scaper')
        ann.duration = duration

        # Add background event
        ann.append(time=0.0,
                   duration=duration,
                   value={
                       "label": "background",
                       "source_file": src_audio_path,
                       "source_time": 0.0,
                       "event_time": 0.0,
                       "event_duration": duration,
                       "snr": 0.0,
                       "role": "background",
                       "pitch_shift": None,
                       "time_stretch": None
                   },
                   confidence=1.0)

        # Create a new event for all of the labels of interest
        for label in LABELS:
            label_id = label_to_id[label]
            presence_key = "{}_{}_presence".format(label_id, label)
            if row[presence_key]:
                ann.append(time=0.0,
                           duration=duration,
                           value={
                               "label": label,
                               "source_file": src_audio_path,
                               "source_time": 0.0,
                               "event_time": 0.0,
                               "event_duration": duration,
                               "snr": 0.0,
                               "role": "foreground",
                               "pitch_shift": None,
                               "time_stretch": None
                           },
                           confidence=1.0)

        # Save JAMS file
        jam.save(dst_jams_path)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    run(annotation_path=args.ust_annotation_path,
        taxonomy_path=args.ust_taxonomy_path,
        data_dir=args.ust_folder,
        out_dir=args.out_folder,
        use_symlinks=args.use_symlinks)

