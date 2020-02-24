import os
import numpy as np
import oyaml as yaml
import pandas as pd
import jams
import torch
import torchaudio
import torchvision
from torch.utils.data import Dataset
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram, MelScale

# Note: if we need to use pescador, see https://github.com/pescadores/pescador/issues/133
# torchaudio.transforms.MelSpectrogram <- Note that this uses "HTK" mels

SAMPLE_RATE = 16000


class CDSDDataset(Dataset):
    def __init__(self, root_dir, subset='train', transform=None):
        subset_names = ('train', 'valid', 'test')
        if subset not in subset_names:
            raise ValueError('Invalid subset: {}'.format(subset))

        self.transform = transform
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, subset)
        self.files = []
        self.subset = subset

        labels = set()
        # Note, subset is being overwritten here
        for subset in subset_names:
            data_dir = os.path.join(root_dir, subset)
            for fname in os.listdir(data_dir):
                # Only look at audio files
                if not fname.endswith('.wav'):
                    continue

                # Make sure there's a corresponding JAMS file
                name = os.path.splitext(fname)[0]
                jams_path = os.path.join(self.root_dir, subset, name + '.jams')
                if not os.path.exists(jams_path):
                    raise ValueError('Missing JAMS file for {} in {}'.format(fname, self.data_dir))

                jams_obj = jams.load(jams_path)
                # Aggregate labels
                for ann in jams_obj.annotations[0].data:
                    if ann.value['role'] == "foreground":
                        labels.add(ann.value['label'])

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

        jams_obj = jams.load(jams_path)

        labels = torch.zeros(self.num_labels)
        for ann in jams_obj.annotations[0].data:
            if ann.value['role'] == "foreground":
                label = ann.value['label']
                idx = self.label_to_idx[label]
                labels[idx] = 1.0

        #audio_data = waveform
        # TODO: TEMPORARY HACK
        audio_data = waveform[0, :64000]
        if self.transform is not None:
            audio_data = self.transform(audio_data)

        sample = {
            'audio_data': audio_data,
            'labels': labels
        }

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
