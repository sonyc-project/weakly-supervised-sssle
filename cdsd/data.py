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
from transforms import Spectrogram, MelSpectrogram, LogMagnitude
from utils import get_torch_window_fn, suppress_stdout, suppress_stderr

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
        self.background_present = False

        # Set audio backend to soundfile
        torchaudio.set_audio_backend('soundfile')

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
                    elif ann.value['role'] == 'background':
                        self.background_present = True

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

    def get_input_shape(self):
        # Assumes all files are the same length
        file = self.files[0]
        audio_path = os.path.join(self.data_dir, file + '.wav')
        waveform_len = int(librosa.get_duration(filename=audio_path) * SAMPLE_RATE)

        is_timefreq = False
        hop_length = 1
        num_bins = None
        if self.transform is not None:
            for t in self.transform.transforms:
                # There should only be at most one transform that
                # effects the time dimension (since we don't allow
                # the resample transformation)
                if isinstance(t, Spectrogram):
                    is_timefreq = True
                    hop_length = t.hop_length
                    num_bins = t.win_length // 2 + 1
                elif isinstance(t, MelSpectrogram):
                    is_timefreq = True
                    hop_length = t.hop_length
                    num_bins = t.n_mels
                elif isinstance(t, MelScale):
                    num_bins = t.n_mels

        if is_timefreq:
            num_frames = waveform_len // hop_length + 1
            return (1, num_bins, num_frames)
        else:
            return (1, waveform_len)

    def get_labels(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        file = self.files[idx]
        audio_path = os.path.join(self.data_dir, file + '.wav')
        jams_path = os.path.join(self.data_dir, file + '.jams')
        waveform_len = int(librosa.get_duration(filename=audio_path) * SAMPLE_RATE)

        is_timefreq = False
        hop_length = 1
        pad_ts = 0.0
        if self.transform is not None:
            for t in self.transform.transforms:
                # There should only be at most one transform that
                # effects the time dimension (since we don't allow
                # the resample transformation)
                if isinstance(t, Spectrogram) or isinstance(t, MelSpectrogram):
                    is_timefreq = True
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
        clip_label_count_arr = torch.zeros(self.num_labels)
        frame_label_count_arr = torch.zeros(num_frames, self.num_labels)
        num_events = 0 #torch.zeros(1, dtype=torch.int16)
        for ann in jams_obj.annotations[0].data:
            if ann.value['role'] == "foreground":
                start_ts = ann.value['event_time']
                end_ts = start_ts + ann.value['event_duration']
                label = ann.value['label']
                label_idx = self.label_to_idx[label]

                # Update clip level labels
                clip_label_arr[label_idx] = 1.0
                clip_label_count_arr[label_idx] += 1.0

                # Update frame level labels
                if is_timefreq:
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
                frame_label_count_arr[start_idx:end_idx, label_idx] += 1.0

                # Increase event count
                num_events += 1

        return {
            'clip_labels': clip_label_arr,
            'frame_labels': frame_label_arr,
            'clip_label_counts': clip_label_count_arr,
            'frame_label_counts': frame_label_count_arr,
            'max_source_polyphony': max_polyphony(frame_label_arr),
            'gini_source_polyphony': gini_polyphony(frame_label_arr),
            'max_event_polyphony': max_polyphony(frame_label_count_arr),
            'max_median_interclass_event_polyphony': max_median_interclass_event_polyphony(frame_label_count_arr),
            'max_median_intraclass_event_polyphony': max_median_intraclass_event_polyphony(frame_label_count_arr),
            'gini_event_polyphony': gini_polyphony(frame_label_count_arr),
            'num_events': num_events,
            'is_timefreq': is_timefreq
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

        # Suppress warnings about wave file headers
        with suppress_stdout():
            with suppress_stderr():
                waveform, sr = torchaudio.load(audio_path)

        if sr != SAMPLE_RATE:
            raise ValueError('Expected sample rate of {} Hz, but got {} Hz ({})'.format(SAMPLE_RATE, sr, audio_path))

        waveform_len = waveform.size()[-1]
        audio_data = waveform
        if self.transform is not None:
            audio_data = self.transform(audio_data)

        sample = self.get_labels(idx)
        is_timefreq = sample.pop('is_timefreq')
        sample['audio_data'] = audio_data
        sample['index'] = torch.tensor(idx, dtype=torch.int16)

        if is_timefreq:
            # Compute energy mask
            frame_energy = audio_data.sum(dim=1, keepdim=True)
            threshold = frame_energy.max(dim=2, keepdim=True)[0] * 0.01
            energy_mask = (frame_energy >= threshold).float()
            sample['energy_mask'] = energy_mask.squeeze()

        if self.load_separation_data:
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
                    label = "background"

                audio_path = os.path.join(event_dir, event_fname)
                # Suppress warnings about wave file headers
                with suppress_stdout():
                    with suppress_stderr():
                        event_waveform, sr = torchaudio.load(audio_path)
                if sr != SAMPLE_RATE:
                    raise ValueError('Expected sample rate of {} Hz, but got {} Hz ({})'.format(SAMPLE_RATE, sr, audio_path))

                if label + '_waveform' not in event_waveforms:
                    event_waveforms[label + '_waveform'] = event_waveform
                else:
                    event_waveforms[label + '_waveform'] += event_waveform

            sample.update(event_waveforms)

            if self.transform is not None:
                for waveform_key, waveform in event_waveforms.items():
                    transformed_key = waveform_key.replace('waveform', 'transformed')
                    sample[transformed_key] = self.transform(waveform)
                    if is_timefreq:
                        # Take element-wise min of source and mixture spectrograms
                        sample[transformed_key] = torch.min(audio_data, sample[transformed_key])

        return sample


def max_polyphony(frame_labels):
    '''
    Given an annotation of sound events, compute the maximum polyphony, i.e.
    the maximum number of simultaneous events at any given point in time. Only
    foreground events are taken into consideration for computing the polyphony.
    Parameters
    '''

    # ([batch], time, label)
    # Get maximum number of simultaneously occurring events
    polyphony = frame_labels.sum(dim=-1).max(dim=-1)[0]
    return polyphony


def max_median_intraclass_event_polyphony(frame_label_counts):
    return torch.median(frame_label_counts, dim=-1)[0].max(dim=-1)[0]


def max_median_interclass_event_polyphony(frame_label_counts):
    # Polyphony defined as the number of events of other classes
    classwise_frame_polyphony = torch.sum(frame_label_counts, dim=-1, keepdim=True) - frame_label_counts
    return torch.median(classwise_frame_polyphony, dim=-1)[0].max(dim=-1)[0]


def gini_polyphony(frame_labels):
    '''
    Compute the gini coefficient of the annotation's polyphony time series.
    Useful as an estimate of the polyphony "flatness" or entropy. The
    coefficient is in the range [0,1] and roughly inverse to entropy: a
    distribution that's close to uniform will have a low gini coefficient
    (high entropy), vice versa.
    https://en.wikipedia.org/wiki/Gini_coefficient
    '''

    # Sample the polyphony using the specified hop size
    values = frame_labels.sum(dim=-1)

    # Compute gini as per:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    values += 1e-6  # all values must be positive
    values = torch.sort(values, dim=-1)[0]  # sort values
    n = values.shape[-1]
    i = torch.arange(n) + 1
    gini = torch.sum((2*i - n - 1) * values, dim=-1) / (n * torch.sum(values, dim=-1))
    return (1 - gini)


def get_data_transforms(train_config):
    input_transform_config = train_config.get("input_transforms", [])
    if len(input_transform_config) > 0:
        transform_list = []
        for transform_config in input_transform_config:
            transform_name = transform_config["name"]
            transform_params = transform_config["parameters"]
            if transform_name == "AmplitudeToDB":
                transform_list.append(AmplitudeToDB(**transform_params))
            elif transform_name == "LogMagnitude":
                transform_list.append(LogMagnitude(**transform_params))
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


def get_spec_params(train_config):
    params = {}
    for transform_config in train_config["input_transforms"]:
        if transform_config["name"] not in ('Spectrogram', 'MelSpectrogram'):
            continue
        params = transform_config["parameters"]
        break
    spec_params = {
        "pad": params.get("pad", 0),
        "n_fft": params.get("n_fft", 400),
        "power": params.get("power", 2.0),
        "normalized": params.get("normalized", False),
        "window_fn": get_torch_window_fn(params.get("window_fn", "hann_window")),
        "window_scaling": params.get("window_scaling", False),
        "wkwargs": params.get("wkwargs", {})
    }
    spec_params["win_length"] = params.get("win_length") or spec_params["n_fft"]
    spec_params["hop_length"] = params.get("hop_length") or (spec_params["win_length"] // 2)
    return spec_params


def get_mel_params(train_config):
    mel_config = None
    for transform_config in train_config["input_transforms"]:
        if transform_config["name"].startswith("Mel"):
            mel_config = {
                "n_mels": transform_config["n_mels"],
                "f_min": transform_config.get("f_min", 0.0),
                "f_max": transform_config.get("f_max", SAMPLE_RATE / 2.0)
            }
            break
    return mel_config


def get_mel_loss_params(train_config):
    mixture_loss_config_list = train_config["losses"].get("mixture", [])
    if not mixture_loss_config_list:
        mixture_loss_config_list = train_config["losses"].get("separation", [])
    mel_params = None
    # Assumes all defined mel params are the same...
    for mixture_loss_config in mixture_loss_config_list:
        curr_mel_params = mixture_loss_config.get("mel_params", None)
        if (curr_mel_params is not None) and (mel_params is not None):
            assert mel_params == curr_mel_params
        mel_params = mel_params or curr_mel_params
    return mel_params

