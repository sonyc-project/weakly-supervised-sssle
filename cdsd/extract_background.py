import os
import numpy as np
import h5py
from cdsd.decrypt import read_encrypted_tar_audio_file
import io
import itertools
import soundfile as sf
import operator


def mask_to_segment_idxs(mask, sample_rate=48000, hop_size=0.5):
    hop_length = int(sample_rate * hop_size)
    segment_idxs_list = []
    for key, it in itertools.groupby(enumerate(mask), key=operator.itemgetter(1)):
        # Skip masked values
        if not key:
            continue

        # Compute start and end sample idxs from frame idxs
        all_frame_idxs = np.array(list(zip(*it))[0])
        start_idx = hop_length * all_frame_idxs[0]
        end_idx = hop_length * (all_frame_idxs[-1] + 1)
        segment_idxs_list.append((start_idx, end_idx))

    return segment_idxs_list


def extract_background_audio(index_path, data_dir, output_dir,
                             cls_version="1.0.0", threshold=0.1, sample_rate=48000,
                             decrypt_url=None, cacert_path=None,
                             cert_path=None, key_path=None):
    year_str = os.path.split(os.path.dirname(index_path))[-1]
    index_name = os.path.splitext(os.path.basename(index_path))[0]
    audio_output_dir = os.path.join(output_dir, year_str, index_name)
    os.makedirs(audio_output_dir, exist_ok=True)

    print("Reading index file.")
    with h5py.File(index_path, 'r') as index_h5:
        timestamp_list = index_h5['recording_index']['timestamp']
        day_hdf5_path_list = index_h5['recording_index']['day_hdf5_path']
        day_hdf5_index_list = index_h5['recording_index']['day_h5_index']

    index_len = len(timestamp_list)
    if index_len == 0:
        print("No available audio. Skipping.")
        return

    print("Reading class predictions file.")
    cls_pred_path = os.path.join(data_dir, 'class_predictions',
                                 cls_version, year_str)
    with h5py.File(cls_pred_path, 'r') as cls_pred_h5:
        cls_pred_len = len(cls_pred_h5['coarse'])
        if cls_pred_len == 0:
            print("No available predictions. Skipping.")
            return

        if cls_pred_len != index_len:
            print("Index length differs from class prediction file length. Skipping.")
            return

        # Get the maximum class prediction for each element
        max_cls_pred_list = None
        for key in cls_pred_h5['coarse'][0].dtype.names:
            if 'timestamp' not in key or 'filename' not in key:
                if max_cls_pred_list is None:
                    max_cls_pred_list = cls_pred_h5['coarse'][key]
                else:
                    max_cls_pred_list = np.maximum(max_cls_pred_list,
                                          cls_pred_h5['coarse'][key])

    print("Grouping index elements by audio filename.")
    # First do an initial pass to group
    day_index_dict = {}
    timestamp_list_dict = {}
    cls_pred_list_dict = {}
    for timestamp, day_hdf5_path, day_hdf5_index, max_cls_pred in zip(timestamp_list, day_hdf5_path_list, day_hdf5_index_list, max_cls_pred_list):
        if day_hdf5_path not in day_index_dict:
            day_index_dict[day_hdf5_path] = []
            timestamp_list_dict[day_hdf5_path] = []
            cls_pred_list_dict[day_hdf5_path] = []
        day_index_dict[day_hdf5_path].append(day_hdf5_index)
        timestamp_list_dict[day_hdf5_path].append(timestamp)
        cls_pred_list_dict[day_hdf5_path].append(max_cls_pred)

    num_files = len(day_index_dict)
    for file_idx, day_hdf5_path in enumerate(day_index_dict.keys()):
        print("* Processing audio for {} ({}/{})".format(day_hdf5_path, file_idx+1, num_files))
        day_index_list = day_index_dict[day_hdf5_path]
        cls_pred_list = cls_pred_list_dict[day_hdf5_path]

        day_hdf5_path = day_hdf5_path.decode()
        audio_hdf5_path = os.path.join(data_dir, day_hdf5_path)

        if len(day_index_list) == 0:
            print("No audio clips for {}, skipping...".format(day_hdf5_path))

        with h5py.File(audio_hdf5_path, 'r') as audio_h5:
            for day_hdf5_index, cls_pred in zip(day_index_list, cls_pred_list):
                if cls_pred >= threshold:
                    continue

                # Read audio
                idx = int(day_hdf5_index)
                filename = audio_h5['recordings'][idx]['filename']
                tar_data = io.BytesIO(audio_h5['recordings'][idx]['data'])
                audio = read_encrypted_tar_audio_file(filename.decode(),
                                                      enc_tar_filebuf=tar_data,
                                                      sample_rate=sample_rate,
                                                      url=decrypt_url,
                                                      cacert=cacert_path,
                                                      cert=cert_path,
                                                      key=key_path)[0]

                # Set up output directory
                day_str = os.path.splitext(os.path.basename(day_hdf5_path))[0]

                fname = "{}_{}.wav".format(day_str, os.path.splitext(filename))
                audio_path = os.path.join(audio_output_dir, fname)

                # Write audio clip
                sf.write(audio_path, audio, samplerate=sample_rate)
                print("Saved {}".format(audio_path))
