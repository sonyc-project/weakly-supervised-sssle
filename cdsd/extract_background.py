import os
import numpy as np
import h5py
from decrypt import read_encrypted_tar_audio_file
import io
import soundfile as sf
import sys
import argparse
from collections import defaultdict
from functools import partial
import multiprocessing
import json
import tqdm


SAMPLE_RATE = 48000


def process_arguments(args):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('index_path', type=str,
                        help='Path to index path or indices folder')

    parser.add_argument('data_folder', type=str,
                        help='Path to root sonyc data folder')

    parser.add_argument('output_folder', type=str,
                        help='Path to folder to output openl3 collections')

    parser.add_argument('--threshold-path', type=str,
                        help='Path to class-wise threshold JSON file')

    parser.add_argument('--default-threshold', type=float,
                        default=0.1,
                        help='Default threshold value for all classes. Ignored if --threshold-path is specified.')

    parser.add_argument('--sensor-fault-threshold', type=float,
                        default=0.43,
                        help='Threshold value for sensor fault detection.')

    parser.add_argument('--cls-version', type=str,
                        default='1.0.0',
                        help='Classifier version to use for class predictions.')

    parser.add_argument('--decrypt-url', type=str,
                        default="https://decrypt-sonyc.engineering.nyu.edu/decrypt",
                        help='SONYC decryption server URL')

    parser.add_argument('--cacert-path', type=str,
                        help='SONYC decryption CA certification path')

    parser.add_argument('--cert-path', type=str,
                        help='SONYC decryption certification path')

    parser.add_argument('--key-path', type=str,
                        help='SONYC decryption key path')

    parser.add_argument('--num-cpus', type=int, default=1,
                        help='Number of cpus to use.')

    return parser.parse_args(args)


def extract_background_audio(index_path, data_dir, output_dir,
                             cls_version="1.0.0", threshold_path=None,
                             default_threshold=0.1, sensor_fault_threshold=0.1,
                             decrypt_url=None, cacert_path=None,
                             cert_path=None, key_path=None):
    year_str = os.path.split(os.path.dirname(index_path))[-1]
    index_name = os.path.splitext(os.path.basename(index_path))[0]
    audio_output_dir = os.path.join(output_dir, year_str, index_name)
    os.makedirs(audio_output_dir, exist_ok=True)

    print("Reading index file: {}".format(index_path))
    with h5py.File(index_path, 'r') as index_h5:
        timestamp_list = index_h5['recording_index']['timestamp']
        day_hdf5_path_list = index_h5['recording_index']['day_hdf5_path']
        day_hdf5_index_list = index_h5['recording_index']['day_h5_index']
        index_len = len(timestamp_list)

        if index_len == 0:
            print("No available audio. Skipping {}\n".format(index_path))
            return

        if 'sensor_fault_aggresive' in index_h5['recording_index'][0].dtype.names:
            sensor_fault_list = index_h5['recording_index']['sensor_fault_aggresive'].max(axis=-1) >= sensor_fault_threshold
        else:
            sensor_fault_list = np.zeros((index_len,)).astype(bool)

    if threshold_path is None:
        threshold_dict = defaultdict(lambda: default_threshold)
    else:
        with open(threshold_path, 'r') as f:
            threshold_dict = json.load(f)

    cls_pred_path = os.path.join(data_dir, 'class_predictions',
                                 cls_version, year_str,
                                 os.path.basename(index_path).replace('recording_index', 'class_predictions'))
    with h5py.File(cls_pred_path, 'r') as cls_pred_h5:
        cls_pred_len = len(cls_pred_h5['coarse'])
        if cls_pred_len == 0:
            print("No class predictions available. Skipping {}\n".format(index_path))
            return

        if cls_pred_len != index_len:
            print("Index length differs from class prediction file length. Skipping {}\n".format(index_path))
            return
        labels = [key for key in cls_pred_h5['coarse'][0].dtype.names
                  if ('timestamp' not in key) and ('filename' not in key)]
        foreground_mask_list = np.zeros((index_len,)).astype(bool)

        # Compute classwise presence using class-specific thresholds and take logical-OR
        for key in labels:
            threshold = threshold_dict[key]
            assert 0 < threshold < 1
            cls_mask_list = cls_pred_h5['coarse'][key] >= threshold
            foreground_mask_list = np.logical_or(foreground_mask_list, cls_mask_list)

        # Negate mask to get background
        background_mask_list = np.logical_not(foreground_mask_list)

    # First do an initial pass to group
    day_index_dict = {}
    timestamp_list_dict = {}
    background_mask_list_dict = {}
    sensor_fault_list_dict = {}
    for timestamp, day_hdf5_path, day_hdf5_index, background_mask, sensor_fault in zip(timestamp_list, day_hdf5_path_list, day_hdf5_index_list, background_mask_list, sensor_fault_list):
        if day_hdf5_path not in day_index_dict:
            day_index_dict[day_hdf5_path] = []
            timestamp_list_dict[day_hdf5_path] = []
            background_mask_list_dict[day_hdf5_path] = []
            sensor_fault_list_dict[day_hdf5_path] = []
        day_index_dict[day_hdf5_path].append(day_hdf5_index)
        timestamp_list_dict[day_hdf5_path].append(timestamp)
        background_mask_list_dict[day_hdf5_path].append(background_mask)
        sensor_fault_list_dict[day_hdf5_path].append(sensor_fault)

    num_files = len(day_index_dict)
    for file_idx, day_hdf5_path in enumerate(day_index_dict.keys()):
        day_index_list = day_index_dict[day_hdf5_path]
        background_mask_list = background_mask_list_dict[day_hdf5_path]
        sensor_fault_list = sensor_fault_list_dict[day_hdf5_path]

        day_hdf5_path = day_hdf5_path.decode()
        audio_hdf5_path = os.path.join(data_dir, day_hdf5_path)

        if len(day_index_list) == 0:
            continue

        with h5py.File(audio_hdf5_path, 'r') as audio_h5:
            for day_hdf5_index, background_mask, sensor_fault in zip(day_index_list, background_mask_list, sensor_fault_list):
                if not background_mask:
                    continue

                if sensor_fault:
                    continue

                # Read audio
                idx = int(day_hdf5_index)
                filename = audio_h5['recordings'][idx]['filename']
                tar_data = io.BytesIO(audio_h5['recordings'][idx]['data'])
                audio = read_encrypted_tar_audio_file(filename.decode(),
                                                      enc_tar_filebuf=tar_data,
                                                      sample_rate=SAMPLE_RATE,
                                                      url=decrypt_url,
                                                      cacert=cacert_path,
                                                      cert=cert_path,
                                                      key=key_path)[0]

                # Set up output directory
                day_str = os.path.splitext(os.path.basename(day_hdf5_path))[0]

                fname = "{}_{}".format(day_str, os.path.basename(filename).decode().replace('.tar.gz', '.wav'))
                audio_path = os.path.join(audio_output_dir, fname)

                # Write audio clip
                sf.write(audio_path, audio, samplerate=SAMPLE_RATE)
                print("Saved {}".format(audio_path))

    print("Finished processing: {}\n".format(index_path))

if __name__ == '__main__':
    params = process_arguments(sys.argv[1:])

    # Set up TF session
    index_path = os.path.abspath(params.index_path)
    worker_func = partial(extract_background_audio,
                          data_dir=os.path.abspath(params.data_folder),
                          output_dir=os.path.abspath(params.output_folder),
                          threshold_path=os.path.abspath(params.threshold_path),
                          default_threshold=params.default_threshold,
                          sensor_fault_threshold=params.sensor_fault_threshold,
                          cls_version=params.cls_version,
                          decrypt_url=params.decrypt_url,
                          cacert_path=params.cacert_path,
                          cert_path=params.cert_path,
                          key_path=params.key_path)

    if os.path.isdir(index_path):
        index_path_list = [os.path.join(root, fname)
                           for root, _, files in os.walk(index_path)
                           for fname in files]
    else:
        assert os.path.isfile(index_path)
        index_path_list = [index_path]

    if params.num_cpus > 1 and len(index_path_list) > 1:
        with multiprocessing.Pool(processes=params.num_cpus) as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(worker_func, index_path_list),
                               total=len(index_path_list)):
                pass
    else:
        for index_path in tqdm.tqdm(index_path_list):
            worker_func(index_path)
