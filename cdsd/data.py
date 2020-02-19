import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset
import jams
from .preprocess_us8k import US8K_TO_SONYCUST_MAP

# Note: if we need to use pescador, see https://github.com/pescadores/pescador/issues/133
# torchaudio.transforms.MelSpectrogram <- Note that this uses "HTK" mels

EXP_DUR = 10.0
SAMPLE_RATE = 16000
LABELS = sorted([x for x in US8K_TO_SONYCUST_MAP.values() if x is not None])
NUM_LABELS = len(LABELS)
LABEL_TO_IDX = {v: k for k, v in enumerate(LABELS)}


class CDSDDataset(Dataset):
    def __init__(self, root_dir, subset='train', transform=None, shuffle=True):
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

        if shuffle:
            random.shuffle(self.files)

    def __len___(self):
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

        target = torch.zeros(NUM_LABELS)
        for ann in jams_obj.annotations:
            if ann.value.role == "foreground":
                label = ann.value.label
                idx = LABEL_TO_IDX[label]
                target[idx] = 1.0

        sample = {
            'waveform': waveform,
            'sample_rate': sr,
            'target': target
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
