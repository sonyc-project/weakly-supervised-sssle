import os
import sys
import pandas as pd
import soundfile as sf
import oyaml as yaml
import jams
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from tqdm import tqdm


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

    return parser.parse_args(args)


def run(annotation_path, taxonomy_path, data_dir, out_dir):
    ann_df = pd.read_csv(annotation_path).sort_values('audio_filename')

    file_list = ann_df['audio_filename'].unique().tolist()

    # Load taxonomy
    with open(taxonomy_path, 'r') as f:
        taxonomy = yaml.load(f)
    labels = [v for v in taxonomy['coarse'].values()]
    label_to_id = {v: k for k, v in taxonomy['coarse'].items()}

    os.makedirs(out_dir, exist_ok=True)

    # Get targets for each file
    for filename in tqdm(file_list):
        # Get all annotation rows pertaining to this file
        file_df = ann_df[ann_df['audio_filename'] == filename]
        # Get the subset split name
        split_str = file_df.iloc[0]['split']
        if split_str == 'validate':
            split_str = 'valid'

        # Deal with input and output paths
        out_subset_dir = os.path.join(out_dir, split_str)
        os.makedirs(out_subset_dir, exist_ok=True)
        src_audio_path = os.path.join(data_dir, split_str, filename)
        dst_audio_path = os.path.join(out_subset_dir, filename)
        dst_jams_path = os.path.join(out_subset_dir, os.path.splitext(filename)[0] + '.jams')

        # Process the file with FFMPEG to ensure that the formats and sample rates are uniform
        cmd_args = ["ffmpeg", "-i", src_audio_path, "-ar", str(SAMPLE_RATE), dst_audio_path]
        res = subprocess.run(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            err_msg = "Error processing {}:\n{}\n{}"
            raise OSError(err_msg.format(src_audio_path, res.stdout, res.stderr))

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

        for label in labels:
            label_id = label_to_id[label]
            presence_key = "{}_{}_presence".format(label_id, label)

            # Get presence across all annotators (unless verified)
            count = 0
            for _, row in file_df.iterrows():
                if int(row['annotator_id']) == 0:
                    # If we have a validated annotation, just use that
                    count = row[presence_key]
                    break
                else:
                    # Otherwise use minority vote
                    count += row[presence_key]

            # If verified positive, or at least one crowdsourced positive, add event
            if count > 0:
                # Add event
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
        out_dir=args.out_folder)

