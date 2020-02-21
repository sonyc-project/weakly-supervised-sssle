import os
import numpy as np
import oyaml as yaml
import pandas as pd
import jams
import torch
import torchaudio
import torchvision
from .preprocess_us8k import US8K_TO_SONYCUST_MAP
from torch.utils.data import Dataset
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram, MelScale
from torch.utils.data import DataLoader

# Note: if we need to use pescador, see https://github.com/pescadores/pescador/issues/133
# torchaudio.transforms.MelSpectrogram <- Note that this uses "HTK" mels

EXP_DUR = 10.0
SAMPLE_RATE = 16000
CDSD_LABELS = sorted([x for x in US8K_TO_SONYCUST_MAP.values() if x is not None])
NUM_CDSD_LABELS = len(CDSD_LABELS)
CDSD_LABEL_TO_IDX = {v: k for k, v in enumerate(CDSD_LABELS)}


class SONYCUSTDataset(Dataset):
    def __init__(self, root_dir, annotation_path, taxonomy_path, subset='train', transform=None):
        if subset not in ('train', 'validate', 'test', 'verified'):
            raise ValueError('Invalid subset: {}'.format(subset))
        self.transform = transform
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, subset)
        self.annotation_path = annotation_path
        self.taxonomy_path = taxonomy_path
        self.subset = subset

        # Load taxonomy and get labels
        with open(self.taxonomy_path, 'r') as f:
            self.taxonomy = yaml.load(f)
        self.labels = [v for v in self.taxonomy['coarse'].values()]
        self.label_keys = ["{}_{}_presence".format(k, v)
                           for k, v in self.taxonomy['coarse'].items()]

        # Load annotations
        ann_df = pd.read_csv(self.annotation_path).sort_values('audio_filename')
        data = ann_df[['split', 'audio_filename']].drop_duplicates().sort_values('audio_filename')

        # Get files for this subset
        self.file_list = []
        for idx, (_, row) in enumerate(data.iterrows()):
            if (subset != 'verified' and row['split'] == subset) or (subset == 'verified' and row['split'] in ('validate', 'test')):
                self.file_list.append(row['audio_filename'])

        # Get targets for each file
        # TODO: Can do this more efficiently without looping twice
        target_list = []
        for filename in self.file_list:
            file_df = ann_df[
                ann_df['audio_filename'] == filename]
            target = []

            for label_key in self.label_keys:
                count = 0

                for _, row in file_df.iterrows():
                    if int(row['annotator_id']) == 0:
                        # If we have a validated annotation, just use that
                        count = row[label_key]
                        break
                    else:
                        count += row[label_key]

                if count > 0:
                    target.append(1.0)
                else:
                    target.append(0.0)

            target_list.append(target)

        self.targets = np.array(target_list)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        fname = self.file_list[idx]
        labels = torch.from_numpy(self.targets[idx])

        audio_path = os.path.join(self.data_dir, fname)
        waveform, sr = torchaudio.load(audio_path)

        if sr != SAMPLE_RATE:
            raise ValueError('Expected sample rate of {} Hz, but got {} Hz ({})'.format(SAMPLE_RATE, sr, audio_path))

        sample = {
            'waveform': waveform,
            'sample_rate': sr,
            'labels': labels
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class CDSDDataset(Dataset):
    def __init__(self, root_dir, subset='train', transform=None):
        if subset not in ('train', 'valid', 'test'):
            raise ValueError('Invalid subset: {}'.format(subset))

        self.transform = transform
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, subset)
        self.files = []
        for fname in os.listdir(self.data_dir):
            # Only look at audio files
            if not fname.endswith('.wav'):
                continue

            name = os.path.splitext(fname)[0]

            # Make sure there's a corresponding JAMS file
            jams_path = os.path.join(self.data_dir, name + '.jams')
            if not os.path.exists(jams_path):
                raise ValueError('Missing JAMS file for {} in {}'.format(fname, self.data_dir))

            self.files.append(name)

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

        jams_obj = jams.load(jams_path)

        labels = torch.zeros(NUM_CDSD_LABELS)
        for ann in jams_obj.annotations:
            if ann.value.role == "foreground":
                label = ann.value.label
                idx = CDSD_LABEL_TO_IDX[label]
                labels[idx] = 1.0

        sample = {
            'waveform': waveform,
            'sample_rate': sr,
            'labels': labels
        }

        if self.transform is not None:
            sample = self.transform(sample)

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
                transform_list.append(MelSpectrogram(sample_rate=SAMPLE_RATE, **transform_params))
            else:
                raise ValueError("Invalid transform type: {}".format(transform_config["name"]))
        return torchvision.transforms.Compose(transform_list)

    return None


def get_batch_input_key(train_config):
    input_key = "waveform"
    for transform_config in train_config.get("input_transform", []):
        transform_name = transform_config["name"]
        if transform_name == "Spectrogram":
            input_key = "specgram"
        elif transform_name == "MelScale" and input_key == "specgram":
            input_key = "mel_specgram"
        elif transform_name == "MelSpectrogram":
            input_key = "mel_specgram"

    return input_key

