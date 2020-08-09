import torch
import torch.nn.functional as F
from functools import partial
from torchaudio.transforms import MelScale
from data import SAMPLE_RATE


def get_mixture_loss_spec_terms(x, labels, masks, energy_mask, energy_masking=None, flatten=True, mel_scale=False, mel_params=None):
    # Optionally apply mel filter bank
    x_orig = x
    mel_tf = None
    if mel_scale:
        if mel_params is None:
            mel_params = {}
        mel_tf = MelScale(sample_rate=SAMPLE_RATE, **mel_params).to(x.device)
        x = mel_tf(x)

    batch_size, n_channel, n_freq, n_time = x.size()
    assert n_channel == 1
    num_labels = labels.size()[-1]

    present_spec = torch.zeros_like(x, device=x.device)
    absent_spec = torch.zeros_like(x, device=x.device)

    if labels.ndim == 2:
        # Broadcast to channel, freq, and time dims
        labels = labels[:, None, None, None, :]
    elif labels.ndim == 3:
        # Broadcast to channel and freq dims
        labels = labels[:, None, None, :, :]
    else:
        raise ValueError('Invalid number of dimensions for labels: {} ({})'.format(labels.ndim, labels.shape))

    for idx in range(num_labels):
        mask = masks[..., idx]
        x_masked = x_orig * mask
        if mel_scale:
            x_masked = mel_tf(x_masked)

        present_spec += x_masked * labels[..., idx]
        absent_spec += x_masked * (1 - labels[..., idx])

    mix_present_spec_diff = x - present_spec

    if energy_masking:
        energy_mask = energy_mask[:, None, None, :]
        mix_present_spec_diff *= energy_mask
        absent_spec *= energy_mask

    if flatten:
        mix_present_spec_diff_flat = mix_present_spec_diff.reshape(batch_size, -1)
        absent_spec_flat = absent_spec.reshape(batch_size, -1)
        return mix_present_spec_diff_flat, absent_spec_flat, x
    else:
        return mix_present_spec_diff, absent_spec, x


def get_normalization_factor(x, energy_mask, energy_masking=None, spectrum=False):
    batch_size, n_channel, n_freq, n_time = x.size()
    if spectrum:
        return torch.tensor(n_freq, dtype=x.dtype, device=x.device)
    elif energy_masking is None:
        return torch.tensor(n_freq * n_time, dtype=x.dtype, device=x.device)
    else:
        return energy_mask.sum(dim=-1) * n_freq


def mixture_loss(x, labels, masks, energy_mask, energy_masking=None, spectrum=False, mel_scale=False):
    if spectrum:
        mix_present_spec_diff, absent_spec, x = get_mixture_loss_spec_terms(x, labels, masks, energy_mask, energy_masking, flatten=False, mel_scale=mel_scale)
        norm_factor = get_normalization_factor(x, energy_mask, spectrum=True)
        # Sum time and channel dimensions
        mix_present_spec_diff = mix_present_spec_diff.sum(dim=-1).sum(dim=1)
        absent_spec = absent_spec.sum(dim=-1).sum(dim=1)
    else:
        mix_present_spec_diff, absent_spec, x = get_mixture_loss_spec_terms(x, labels, masks, energy_mask, energy_masking, flatten=True, mel_scale=mel_scale)
        norm_factor = get_normalization_factor(x, energy_mask, energy_masking=energy_masking)

    present_energy_diff = torch.norm(mix_present_spec_diff, p=1, dim=1) / norm_factor
    absent_energy = torch.norm(absent_spec, p=1, dim=1) / norm_factor

    mix_loss = (present_energy_diff + absent_energy).mean()
    return mix_loss


def mixture_margin_loss(x, labels, masks, energy_mask, energy_masking=None, margin=None, spectrum=False, mel_scale=False):
    assert margin is not None
    if spectrum:
        mix_present_spec_diff, absent_spec, x = get_mixture_loss_spec_terms(x, labels, masks, energy_mask, energy_masking, flatten=False, mel_scale=mel_scale)
        norm_factor = get_normalization_factor(x, energy_mask, spectrum=True)
        # Sum time and channel dimensions
        mix_present_spec_diff = mix_present_spec_diff.sum(dim=-1).sum(dim=1)
        absent_spec = absent_spec.sum(dim=-1).sum(dim=1)
    else:
        mix_present_spec_diff, absent_spec, x = get_mixture_loss_spec_terms(x, labels, masks, energy_mask, energy_masking, flatten=True, mel_scale=mel_scale)
        norm_factor = get_normalization_factor(x, energy_mask, energy_masking=energy_masking)

    present_loss = F.relu(torch.norm(mix_present_spec_diff, p=1, dim=1) - margin) / norm_factor
    absent_loss = torch.norm(absent_spec, p=1, dim=1) / norm_factor
    mix_loss = (present_loss + absent_loss).mean()
    return mix_loss


def mixture_margin_asymmetric_loss(x, labels, masks, energy_mask, energy_masking=None, margin=None, spectrum=False, mel_scale=False):
    assert margin is not None
    if spectrum:
        mix_present_spec_diff, absent_spec, x = get_mixture_loss_spec_terms(x, labels, masks, energy_mask, energy_masking, flatten=False, mel_scale=mel_scale)
        norm_factor = get_normalization_factor(x, energy_mask, spectrum=True)
        # Sum time and channel dimensions
        mix_present_spec_diff = mix_present_spec_diff.sum(dim=-1).sum(dim=1)
        absent_spec = absent_spec.sum(dim=-1).sum(dim=1)
    else:
        mix_present_spec_diff, absent_spec, x = get_mixture_loss_spec_terms(x, labels, masks, energy_mask, energy_masking, flatten=True, mel_scale=mel_scale)
        norm_factor = get_normalization_factor(x, energy_mask, energy_masking=energy_masking)

    present_underest = F.relu(F.relu(mix_present_spec_diff).sum(dim=1) - margin) / norm_factor
    present_overest = F.relu(-mix_present_spec_diff).sum(dim=1) / norm_factor
    present_term = present_underest + present_overest
    absent_energy = torch.norm(absent_spec, p=1, dim=1) / norm_factor

    mix_loss = (present_term + absent_energy).mean()
    return mix_loss


def get_mixture_loss_function(train_config):
    mixture_loss_config = train_config["losses"]["mixture"]
    loss_name = mixture_loss_config["name"]
    energy_masking = mixture_loss_config.get("energy_masking", False)
    spectrum = mixture_loss_config.get("spectrum", False)
    mel_scale = mixture_loss_config.get("mel_scale", False)

    if loss_name == "mixture_loss":
        return partial(mixture_loss,
                       energy_masking=energy_masking,
                       spectrum=spectrum,
                       mel_scale=mel_scale)
    elif loss_name == "mixture_margin_loss":
        return partial(mixture_margin_loss, margin=mixture_loss_config["margin"],
                       energy_masking=energy_masking,
                       spectrum=spectrum,
                       mel_scale=mel_scale)
    elif loss_name == "mixture_margin_asymmetric_loss":
        return partial(mixture_margin_asymmetric_loss,
                       margin=mixture_loss_config["margin"],
                       energy_masking=energy_masking,
                       spectrum=spectrum,
                       mel_scale=mel_scale)
    else:
        raise ValueError("Invalid mixture loss type: {}".format(mixture_loss_config["name"]))

