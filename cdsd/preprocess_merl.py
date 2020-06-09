import os
import json
import shutil
import sys
import librosa
import jams
from argparse import ArgumentParser, RawDescriptionHelpFormatter


SAMPLE_RATE = 16000


def parse_arguments(args):
    parser = ArgumentParser(sys.argv[0],
                            description="Preprocess MERL mixtures to be in same format as synthesized mixtures.",
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('data_folder', type=str,
                        help='Path to dataset.')
    parser.add_argument('out_folder', type=str,
                        help='Output directory where audio files will be saved.')

    return parser.parse_args(args)


def run(data_dir, out_dir):
    prefix = "pishdadian2020merl-lambda5"
    for subset in ('train', 'test', 'valid'):
        src_subset_dir = os.path.join(data_dir, subset)
        dst_subset_dir = os.path.join(out_dir, subset)
        metadata_dir = os.path.join(src_subset_dir, "metadata")
        mixture_dir = os.path.join(src_subset_dir, "mix")

        os.makedirs(dst_subset_dir, exist_ok=True)

        for fname in os.listdir(metadata_dir):
            metadata_path = os.path.join(metadata_dir, fname)
            src_ex_name = os.path.splitext(fname)[0]
            num = src_ex_name.lstrip("0")
            if not num:
                num = "0"

            dst_ex_name = prefix + "_" + num
            src_audio_fname = src_ex_name + '.wav'
            dst_audio_fname = dst_ex_name + '.wav'
            src_mixture_path = os.path.join(mixture_dir, src_audio_fname)

            # Get audio duration
            duration = librosa.get_duration(filename=src_mixture_path)

            # Set up output paths
            dst_mixture_path = os.path.join(dst_subset_dir, dst_audio_fname)
            jams_path = os.path.join(dst_subset_dir, dst_ex_name + '.jams')
            events_dir = os.path.join(dst_subset_dir, dst_ex_name + "_events")
            os.makedirs(events_dir, exist_ok=True)

            # Copy mixture
            shutil.copy(src_mixture_path, dst_mixture_path)

            with open(metadata_path, 'r') as f:
                event_list = json.load(f)

            present_labels = set()
            # Create JAMS
            jam = jams.JAMS()
            jam.file_metadata.duration = duration
            ann = jams.Annotation(namespace='scaper')
            ann.duration = duration
            for event in event_list:
                label = event["class"]
                start_ts = event["start_sample"] / SAMPLE_RATE
                end_ts = event["end_sample"] / SAMPLE_RATE
                src_fname = os.path.basename(event["filepath"])
                gain = event["gain"]

                present_labels.add(label)

                # Create annotation
                ann.append(time=0.0,
                           duration=duration,
                           value={
                               "label": label,
                               "source_file": src_fname,
                               "source_time": 0.0,
                               "event_time": start_ts,
                               "event_duration": end_ts - start_ts,
                               "snr": gain,
                               "role": "foreground",
                               "pitch_shift": None,
                               "time_stretch": None
                           },
                           confidence=1.0)

            jam.annotations.append(ann)
            jam.save(jams_path)

            # Copy separated sources
            for idx, label in enumerate(present_labels):
                label_dir = os.path.join(src_subset_dir, "s_" + label)
                src_source_path = os.path.join(label_dir, src_audio_fname)
                dst_source_fname = "foreground{}_{}.wav".format(idx, label)
                dst_source_path = os.path.join(events_dir, dst_source_fname)
                shutil.copy(src_source_path, dst_source_path)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    run(data_dir=args.data_folder,
        out_dir=args.out_folder)

