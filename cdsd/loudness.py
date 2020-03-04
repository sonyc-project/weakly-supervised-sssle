# Adapted from https://github.com/sonyc-project/sonycnode/blob/79fa47652b37c47a7e289f183613ec02fdb9abbe/capture/weight_filts.py
import numpy as np
import torch
from torchaudio.functional import complex_norm


def get_freq_weighting(nfft, sr, weighting='a', device=None):
    # Create array of frequency bin values for octave band segmentation
    # JTC: Need to make double precision to prevent overflow
    amp_array = torch.linspace(0, sr / 2.0, steps=1 + nfft // 2, dtype=torch.float64)
    if device is not None:
        amp_array = amp_array.to(device)

    # Set offset on any zero values to remove log(0) situations
    amp_array[amp_array == 0] = 1e-17
    amp_array = torch.pow(amp_array, 2)

    if weighting == 'a':
        # Constant values for A and C weighting calculation
        c1 = 3.5041384e16
        c2 = 20.598997 ** 2
        c3 = 107.65265 ** 2
        c4 = 737.86223 ** 2
        c5 = 12194.217 ** 2

        # A weighting amplitude calculation
        num = c1 * torch.pow(amp_array, 4)
        den = torch.pow(c2 + amp_array, 2) * (c3 + amp_array) * (c4 + amp_array) * torch.pow(c5 + amp_array, 2)
    elif weighting == 'c':
        # Constant values for A and C weighting calculation
        c1 = 2.242881e16
        c2 = 20.598997 ** 2
        c5 = 12194.217 ** 2

        # C weighting amplitude calculation
        num = c1 * torch.pow(amp_array, 2)
        den = torch.pow(c2 + amp_array, 2) * torch.pow(c5 + amp_array, 2)
    else:
        raise ValueError('Invalid weighting type: {}'.format(weighting))

    weighting_array = num / den
    # Cast back to single precision
    return weighting_array.float()


def compute_dbfs(audio, sr, weighting='a', device=None):
    audio_len = audio.size()[-1]
    window = torch.hann_window(audio_len)[None, :]
    if device is not None:
        window = window.to(device)
    # FFT input buffer using appropriate FFT size for octave band analysis
    sp = complex_norm(torch.rfft(audio * window, 1))
    sp[sp == 0] = 1e-17
    sp = torch.pow(sp, 2)

    weighting = get_freq_weighting(audio_len, sr, weighting=weighting, device=device)[None, :]
    sp = weighting * sp

    mean_energy = torch.sum(sp, dim=-1) / ((1.0 / sr) * audio_len)
    dbfs = 10 * torch.log10(mean_energy)
    return dbfs
