# Adapted from https://github.com/sonyc-project/sonycnode/blob/79fa47652b37c47a7e289f183613ec02fdb9abbe/capture/weight_filts.py
import torch
from torch.nn import Module
from torchaudio.functional import complex_norm, create_fb_matrix
from data import get_spec_params, get_mel_params, get_mel_loss_params


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


def compute_dbfs(audio, sr, train_config, weighting='a', device=None):
    n_fft = audio.size()[-1]
    spec_params = get_spec_params(train_config)
    window = spec_params["window_fn"](n_fft)[None, :]
    if device is not None:
        window = window.to(device)
    # FFT input buffer using appropriate FFT size for octave band analysis
    spec = complex_norm(torch.rfft(audio * window, 1))
    spec[spec == 0] = 1e-17
    spec = torch.pow(spec, 2)
    spec = spec.squeeze(dim=1)

    weighting = get_freq_weighting(n_fft, sr, weighting=weighting, device=device)[None, :]
    spec = weighting * spec

    # Account for DC/Nyquist
    spec[..., 0] *= 0.5
    if n_fft % 2 == 0:
        spec[..., -1] *= 0.5

    scale = 2.0 / (n_fft ** 2)
    mean_energy = scale * spec.sum(dim=-1)
    dbfs = 10 * torch.log10(mean_energy)
    return dbfs


def compute_dbfs_spec(spec, sr, spec_params, mel_scale=False, mel_params=None, weighting='a', device=None):
    n_fft = spec_params["n_fft"]

    # Account for window scaling
    if spec_params["window_scaling"]:
        spec = spec * spec_params["window_fn"](n_fft).sum()

    # Squeeze channel dim
    spec = spec.squeeze(dim=1)
    spec[spec == 0] = 1e-17
    spec = torch.pow(spec, 2)
    # Take mean of frequency bins across time
    spec = spec.mean(dim=-1)

    weighting = get_freq_weighting(n_fft, sr, weighting=weighting, device=device)
    # If we are estimating in the mel frequency scale, alter freq band weights
    if mel_scale:
        assert mel_params is not None
        fb = create_fb_matrix(n_freqs=n_fft // 2 + 1,
                              f_min=mel_params.get("f_min", 0.0),
                              f_max=mel_params.get("f_max", sr / 2.0),
                              n_mels=mel_params["n_mels"],
                              sample_rate=sr)
        if device is not None:
            fb = fb.to(device)
        weighting = torch.matmul(fb.T, weighting)
    spec = spec * weighting.unsqueeze(0)

    # Account for DC/Nyquist
    spec[..., 0] *= 0.5
    if n_fft % 2 == 0:
        spec[..., -1] *= 0.5

    scale = 2.0 / (n_fft ** 2)
    mean_energy = scale * spec.sum(dim=-1)
    dbfs = 10 * torch.log10(mean_energy)
    return dbfs

