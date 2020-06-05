import numpy as np
import os
import warnings
import jams
import librosa
import torch
import torchaudio
import torchvision
from torch.utils.data import Dataset
from torchaudio.transforms import AmplitudeToDB, MelScale
from spectrum import Spectrogram, MelSpectrogram
from utils import get_torch_window_fn

# Note: if we need to use pescador, see https://github.com/pescadores/pescador/issues/133
# torchaudio.transforms.MelSpectrogram <- Note that this uses "HTK" mels

SAMPLE_RATE = 16000


class CDSDDataset(Dataset):
    def __init__(self, root_dir, subset='train', transform=None,
                 load_separation_data=False):
        subset_names = ('train', 'valid', 'test')
        if subset not in subset_names:
            raise ValueError('Invalid subset: {}'.format(subset))

        self.transform = transform
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, subset)
        self.files = []
        self.subset = subset
        self.load_separation_data = load_separation_data

        labels = set()
        # Note, subset is being overwritten here
        for subset in subset_names:
            data_dir = os.path.join(root_dir, subset)
            if not os.path.isdir(data_dir):
                warnings.warn('Could not find subset {} in {}. Skipping.'.format(subset, root_dir))
                continue

            for fname in os.listdir(data_dir):
                # Only look at audio files
                if not fname.endswith('.wav'):
                    continue
                name = os.path.splitext(fname)[0]

                # Make sure there's a corresponding JAMS file
                jams_path = os.path.join(self.root_dir, subset, name + '.jams')
                if not os.path.exists(jams_path):
                    raise ValueError('Missing JAMS file for {} in {}'.format(fname, self.data_dir))

                jams_obj = jams.load(jams_path)
                # Aggregate labels
                for ann in jams_obj.annotations[0].data:
                    if ann.value['role'] == "foreground":
                        labels.add(ann.value['label'])

                # If we are loading the separation data, only include the files
                # that have separated sources
                if load_separation_data:
                    event_dir = os.path.join(data_dir, name + "_events")
                    if not os.path.isdir(event_dir):
                        continue

                if subset == self.subset:
                    self.files.append(name)

        self.files = sorted(self.files)
        self.labels = sorted(labels)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.num_labels = len(self.labels)

        total_frames = 0
        frame_label_counts = torch.zeros(self.num_labels)
        for idx in range(len(self.files)):
            frame_labels = self.get_labels(idx)['frame_labels']
            total_frames += frame_labels.size()[0]
            frame_label_counts += frame_labels.sum(dim=0)
        total_frames = torch.tensor(total_frames, dtype=torch.float32)
        self.class_frame_priors = frame_label_counts / total_frames

    def __len__(self):
        return len(self.files)

    def get_labels(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        file = self.files[idx]
        audio_path = os.path.join(self.data_dir, file + '.wav')
        jams_path = os.path.join(self.data_dir, file + '.jams')
        waveform_len = int(librosa.get_duration(filename=audio_path) * SAMPLE_RATE)

        is_stft = False
        hop_length = 1
        pad_ts = 0.0
        if self.transform is not None:
            for t in self.transform.transforms:
                # There should only be at most one transform that
                # effects the time dimension (since we don't allow
                # the resample transformation)
                if isinstance(t, Spectrogram) or isinstance(t, MelSpectrogram):
                    is_stft = True
                    hop_length = t.hop_length
                    # Account for centering
                    pad_ts = (t.win_length // 2) / SAMPLE_RATE
                    break
        hop_ts = hop_length / SAMPLE_RATE
        # Assumes centering
        num_frames = waveform_len // hop_length + 1

        jams_obj = jams.load(jams_path)

        # Compute clip and frame level labels
        clip_label_arr = torch.zeros(self.num_labels)
        frame_label_arr = torch.zeros(num_frames, self.num_labels)
        num_events = torch.zeros(1, dtype=torch.int16)
        for ann in jams_obj.annotations[0].data:
            if ann.value['role'] == "foreground":
                start_ts = ann.value['event_time']
                end_ts = start_ts + ann.value['event_duration']
                label = ann.value['label']
                label_idx = self.label_to_idx[label]

                # Update clip level labels
                clip_label_arr[label_idx] = 1.0

                # Update frame level labels
                if is_stft:
                    # Compute frame indices, such that each frame contains
                    # the source
                    start_idx = int((start_ts - pad_ts + hop_ts) * SAMPLE_RATE / hop_length)
                    start_idx = max(0, start_idx)
                    end_idx = int(np.ceil((end_ts + pad_ts) * SAMPLE_RATE / hop_length))
                    end_idx = min(num_frames, end_idx)
                else:
                    start_idx = int(start_ts * SAMPLE_RATE)
                    end_idx = int(np.ceil(end_ts * SAMPLE_RATE))
                frame_label_arr[start_idx:end_idx, label_idx] = 1.0

                # Increase event count
                num_events += 1

        return {
            'clip_labels': clip_label_arr,
            'frame_labels': frame_label_arr,
            'num_events': num_events,
            'is_stft': is_stft
        }

    def get_num_frames(self):
        file = self.files[0]
        audio_path = os.path.join(self.data_dir, file + '.wav')
        waveform_len = int(librosa.get_duration(filename=audio_path) * SAMPLE_RATE)

        hop_length = 1
        if self.transform is not None:
            for t in self.transform.transforms:
                if isinstance(t, Spectrogram) or isinstance(t, MelSpectrogram):
                    hop_length = t.hop_length
                    break
        # Assumes centering
        num_frames = waveform_len // hop_length + 1

        return num_frames

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        file = self.files[idx]
        audio_path = os.path.join(self.data_dir, file + '.wav')

        waveform, sr = torchaudio.load(audio_path)

        if sr != SAMPLE_RATE:
            raise ValueError('Expected sample rate of {} Hz, but got {} Hz ({})'.format(SAMPLE_RATE, sr, audio_path))

        waveform_len = waveform.size()[-1]
        audio_data = waveform
        if self.transform is not None:
            audio_data = self.transform(audio_data)

        labels_dict = self.get_labels(idx)
        is_stft = labels_dict['is_stft']

        sample = {
            'audio_data': audio_data,
            'clip_labels': labels_dict['clip_labels'],
            'frame_labels': labels_dict['frame_labels'],
            'index': torch.tensor(idx, dtype=torch.int16)
        }

        if is_stft:
            # Compute energy mask
            frame_energy = audio_data.sum(dim=1, keepdim=True)
            threshold = frame_energy.max(dim=2, keepdim=True)[0] * 0.01
            energy_mask = (frame_energy >= threshold).float()
            sample['energy_mask'] = energy_mask.squeeze()

        if self.load_separation_data:
            sample['num_events'] = labels_dict['num_events']

            # Include mixture and separated source waveforms
            sample['mixture_waveform'] = waveform

            event_dir = os.path.join(self.data_dir, file + "_events")
            if not os.path.isdir(event_dir):
                print("Could not find events for {} in {}".format(file, self.data_dir))

            event_waveforms = {label + "_waveform": torch.zeros(1, waveform_len) for label in self.labels}
            # Accumulate event waveforms by label
            for event_fname in os.listdir(event_dir):
                name = os.path.splitext(event_fname)[0]
                split_idx = name.index('_')
                prefix = name[:split_idx]
                label = name[split_idx+1:]

                # If if background is one of the labels, accumulate separately
                if prefix.startswith('background'):
                    if "background" in self.labels:
                        label = "background"
                    else:
                        continue

                audio_path = os.path.join(event_dir, event_fname)
                event_waveform, sr = torchaudio.load(audio_path)
                if sr != SAMPLE_RATE:
                    raise ValueError('Expected sample rate of {} Hz, but got {} Hz ({})'.format(SAMPLE_RATE, sr, audio_path))

                event_waveforms[label + '_waveform'] += event_waveform

            sample.update(event_waveforms)

            if self.transform is not None:
                for label in self.labels:
                    sample[label + '_transformed'] = self.transform(event_waveforms[label + '_waveform'])

        return sample


def get_data_transforms(train_config):
    input_transform_config = train_config.get("input_transforms", [])
    if len(input_transform_config) > 0:
        transform_list = []
        for transform_config in input_transform_config:
            transform_name = transform_config["name"]
            transform_params = transform_config["parameters"]
            if transform_name == "AmplitudeToDB":
                transform_list.append(AmplitudeToDB(**transform_params))
            elif transform_name == "MelScale":
                transform_list.append(MelScale(sample_rate=SAMPLE_RATE, **transform_params))
            elif transform_name == "MelSpectrogram":
                transform_params = dict(transform_params)
                window_params = transform_params.pop('wkwargs', {})
                window_fn = get_torch_window_fn(transform_params.pop('window_fn', 'hann_window'))
                transform_list.append(MelSpectrogram(sample_rate=SAMPLE_RATE,
                                                     window_fn=window_fn,
                                                     wkwargs=window_params,
                                                     **transform_params))
            elif transform_name == "Spectrogram":
                transform_params = dict(transform_params)
                window_params = transform_params.pop('wkwargs', {})
                window_fn = get_torch_window_fn(transform_params.pop('window_fn', 'hann_window'))
                transform_list.append(Spectrogram(window_fn=window_fn,
                                                  wkwargs=window_params,
                                                  **transform_params))
            else:
                raise ValueError("Invalid transform type: {}".format(transform_config["name"]))
        return torchvision.transforms.Compose(transform_list)

    return None
