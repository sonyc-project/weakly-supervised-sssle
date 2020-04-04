import numpy as np
import os
import warnings
import jams
import torch
import torchaudio
import torchvision
from torch.utils.data import Dataset
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram, MelScale, Spectrogram
from utils import get_torch_window_fn

# Note: if we need to use pescador, see https://github.com/pescadores/pescador/issues/133
# torchaudio.transforms.MelSpectrogram <- Note that this uses "HTK" mels

SAMPLE_RATE = 16000


class CDSDDataset(Dataset):
    def __init__(self, root_dir, subset='train', transform=None,
                 load_separation_data=False, label_mode='clip'):
        subset_names = ('train', 'valid', 'test')
        if subset not in subset_names:
            raise ValueError('Invalid subset: {}'.format(subset))

        if label_mode not in ('clip', 'frame'):
            raise ValueError('Invalid label mode: {}'.format(label_mode))

        self.transform = transform
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, subset)
        self.files = []
        self.subset = subset
        self.load_separation_data = load_separation_data
        self.label_mode = label_mode

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
                    event_dir = os.path.join(self.data_dir, name + "_events")
                    if not os.path.isdir(event_dir):
                        continue

                if subset == self.subset:
                    self.files.append(name)

        self.labels = sorted(labels)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.num_labels = len(self.labels)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        file = self.files[idx]
        audio_path = os.path.join(self.data_dir, file + '.wav')
        jams_path = os.path.join(self.data_dir, file + '.jams')

        waveform, sr = torchaudio.load(audio_path)

        if sr != SAMPLE_RATE:
            raise ValueError('Expected sample rate of {} Hz, but got {} Hz ({})'.format(SAMPLE_RATE, sr, audio_path))

        waveform_len = waveform.size()[-1]
        audio_data = waveform
        if self.transform is not None:
            audio_data = self.transform(audio_data)
        jams_obj = jams.load(jams_path)

        if self.label_mode == 'clip':
            label_arr = torch.zeros(self.num_labels)
            num_events = torch.zeros(1, dtype=torch.int16)
            for ann in jams_obj.annotations[0].data:
                if ann.value['role'] == "foreground":
                    label = ann.value['label']
                    label_idx = self.label_to_idx[label]
                    label_arr[label_idx] = 1.0
                    num_events += 1
        elif self.label_mode == 'frame':
            hop_length = 1
            win_length = 1
            is_stft = False

            if self.transform is not None:
                for t in self.transform.transforms:
                    # There should only be at most one transform that
                    # effects the time dimension (since we don't allow
                    # the resample transformation)
                    if isinstance(t, Spectrogram) or isinstance(MelSpectrogram):
                        hop_length = t.hop_length
                        win_length = t.win_length
                        is_stft = True

            # Account for centering
            if is_stft:
                pad_ts = (win_length // 2) / SAMPLE_RATE
            else:
                pad_ts = 0.0

            hop_ts = hop_length / SAMPLE_RATE
            num_frames = audio_data.size()[-1]

            label_arr = torch.zeros(num_frames, self.num_labels)
            num_events = torch.zeros(1, dtype=torch.int16)
            for ann in jams_obj.annotations[0].data:
                if ann.value['role'] == "foreground":
                    start_ts = ann.value['event_time']
                    end_ts = start_ts + ann.value['event_duration']

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

                    label = ann.value['label']
                    label_idx = self.label_to_idx[label]
                    label_arr[start_idx:end_idx, label_idx] = 1.0
                    num_events += 1

        else:
            raise ValueError('Invalid label mode: {}'.format(self.label_mode))

        sample = {
            'audio_data': audio_data,
            'labels': label_arr,
            'index': torch.tensor(idx, dtype=torch.int16)
        }

        if self.load_separation_data:
            sample['num_events'] = num_events

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
