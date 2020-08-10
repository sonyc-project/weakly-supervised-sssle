import torch
import torch.nn.functional as F
from functools import partial
from torchaudio.transforms import MelScale
from data import SAMPLE_RATE, get_spec_params
from loudness import compute_dbfs_spec


def get_mixture_loss_spec_terms(x, labels, masks, energy_mask, energy_masking=False, flatten=True, mel_scale=False, mel_params=None):
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
        present_spec *= energy_mask

    if flatten:
        mix_present_spec_diff_flat = mix_present_spec_diff.reshape(batch_size, -1)
        absent_spec_flat = absent_spec.reshape(batch_size, -1)
        present_spec_flat = present_spec.reshape(batch_size, -1)
        return mix_present_spec_diff_flat, present_spec_flat, absent_spec_flat, x
    else:
        return mix_present_spec_diff, present_spec, absent_spec, x


def get_normalization_factor(x, energy_masking=False, target_type="timefreq"):
    batch_size, n_channel, n_freq, n_time = x.size()
    if target_type == "timefreq" and energy_masking is None:
        return torch.tensor(n_freq * n_time, dtype=x.dtype, device=x.device)
    elif target_type == "timefreq" and energy_masking is not None:
        return torch.tensor(n_freq * n_time, dtype=x.dtype, device=x.device)
    elif target_type == "spectrum":
        return torch.tensor(n_freq, dtype=x.dtype, device=x.device)
    elif target_type in ("dbfs", "energy"):
        return torch.tensor(1, dtype=x.dtype, device=x.device)
    else:
        raise ValueError("Invalid target type: {}".format(target_type))


def transform_spec_to_target(mix_present_spec_diff, present_spec, absent_spec, x, target_type, spec_params=None, mel_params=None):
    batch_size, n_channel, n_freq, n_time = x.size()
    if target_type == "timefreq":
        mix_present_spec_diff = mix_present_spec_diff.reshape(batch_size, -1)
        absent_spec = absent_spec.reshape(batch_size, -1)
    elif target_type == "spectrum":
        # Sum time and channel dimensions
        mix_present_spec_diff = mix_present_spec_diff.sum(dim=-1).sum(dim=1)
        absent_spec = absent_spec.sum(dim=-1).sum(dim=1)
    elif target_type == "dbfs":
        x_dbfs = compute_dbfs_spec(x, SAMPLE_RATE, spec_params=spec_params, mel_params=mel_params, device=x.device)
        present_dbfs = compute_dbfs_spec(present_spec, SAMPLE_RATE, spec_params=spec_params, mel_params=mel_params, device=x.device)
        mix_present_spec_diff = (x_dbfs - present_dbfs).unsqueeze(-1)
        absent_spec = compute_dbfs_spec(absent_spec, SAMPLE_RATE, spec_params=spec_params, mel_params=mel_params, device=x.device).unsqueeze(-1)
    elif target_type == "energy":
        # Sum time, freq, and channel dimensions
        mix_present_spec_diff = mix_present_spec_diff.sum(dim=-1).sum(dim=-1).sum(dim=-1, keepdim=True)
        absent_spec = absent_spec.sum(dim=-1).sum(dim=-1).sum(dim=-1, keepdim=True)
    else:
        raise ValueError("Invalid target type: {}".format(target_type))

    return mix_present_spec_diff, absent_spec


def mixture_loss(x, labels, masks, energy_mask, energy_masking=False, target_type="timefreq", spec_params=None, mel_scale=False, mel_params=None):
    mix_present_spec_diff, present_spec, absent_spec, x = get_mixture_loss_spec_terms(x, labels, masks, energy_mask, energy_masking, flatten=(target_type == "timefreq"), mel_scale=mel_scale, mel_params=mel_params)
    mix_present_spec_diff, absent_spec = transform_spec_to_target(mix_present_spec_diff, present_spec, absent_spec, x, target_type, spec_params=spec_params, mel_params=mel_params)
    norm_factor = get_normalization_factor(x, energy_masking=energy_masking, target_type=target_type)

    present_loss = torch.norm(mix_present_spec_diff, p=1, dim=1) / norm_factor
    # Don't apply norm when using dBFS
    if target_type != "dbfs":
        absent_loss = torch.norm(absent_spec, p=1, dim=1) / norm_factor
    else:
        absent_loss = absent_spec.squeeze(-1)

    mix_loss = (present_loss + absent_loss).mean()
    return mix_loss


def mixture_margin_loss(x, labels, masks, energy_mask, energy_masking=False, margin=None, target_type="timefreq", spec_params=None, mel_scale=False, mel_params=None):
    assert margin is not None
    mix_present_spec_diff, present_spec, absent_spec, x = get_mixture_loss_spec_terms(x, labels, masks, energy_mask, energy_masking, flatten=(target_type == "timefreq"), mel_scale=mel_scale, mel_params=mel_params)
    mix_present_spec_diff, absent_spec = transform_spec_to_target(mix_present_spec_diff, present_spec, absent_spec, x, target_type, spec_params=spec_params, mel_params=mel_params)
    norm_factor = get_normalization_factor(x, energy_masking=energy_masking, target_type=target_type)

    present_loss = F.relu(torch.norm(mix_present_spec_diff, p=1, dim=1) - margin) / norm_factor
    # Don't apply norm when using dBFS
    if target_type != "dbfs":
        absent_loss = torch.norm(absent_spec, p=1, dim=1) / norm_factor
    else:
        absent_loss = absent_spec.squeeze(-1)

    mix_loss = (present_loss + absent_loss).mean()
    return mix_loss


def mixture_margin_asymmetric_loss(x, labels, masks, energy_mask, energy_masking=False, margin=None, target_type="timefreq", spec_params=None, mel_scale=False, mel_params=None):
    assert margin is not None
    mix_present_spec_diff, present_spec, absent_spec, x = get_mixture_loss_spec_terms(x, labels, masks, energy_mask, energy_masking, flatten=(target_type == "timefreq"), mel_scale=mel_scale, mel_params=mel_params)
    mix_present_spec_diff, absent_spec = transform_spec_to_target(mix_present_spec_diff, present_spec, absent_spec, x, target_type, spec_params=spec_params, mel_params=mel_params)
    norm_factor = get_normalization_factor(x, energy_masking=energy_masking, target_type=target_type)

    present_underest = F.relu(F.relu(mix_present_spec_diff).sum(dim=1) - margin) / norm_factor
    present_overest = F.relu(-mix_present_spec_diff).sum(dim=1) / norm_factor
    present_loss = present_underest + present_overest
    # Don't apply norm when using dBFS
    if target_type != "dbfs":
        absent_loss = torch.norm(absent_spec, p=1, dim=1) / norm_factor
    else:
        absent_loss = absent_spec.squeeze(-1)

    mix_loss = (present_loss + absent_loss).mean()
    return mix_loss


def get_mixture_loss_function(train_config):
    mixture_loss_config = train_config["losses"]["mixture"]
    loss_name = mixture_loss_config["name"]
    energy_masking = mixture_loss_config.get("energy_masking", False)
    target_type = mixture_loss_config.get("target_type", "timefreq")
    spec_params = get_spec_params(train_config)
    mel_scale = mixture_loss_config.get("mel_scale", False)
    mel_params = mixture_loss_config.get("mel_params", None)
    assert ((not mel_scale) and (mel_params is None)) or (mel_scale and (mel_params is not None))

    if loss_name == "mixture_loss":
        return partial(mixture_loss,
                       energy_masking=energy_masking,
                       target_type=target_type,
                       spec_params=spec_params,
                       mel_scale=mel_scale,
                       mel_params=mel_params)
    elif loss_name == "mixture_margin_loss":
        return partial(mixture_margin_loss, margin=mixture_loss_config["margin"],
                       energy_masking=energy_masking,
                       target_type=target_type,
                       spec_params=spec_params,
                       mel_scale=mel_scale,
                       mel_params=mel_params)
    elif loss_name == "mixture_margin_asymmetric_loss":
        return partial(mixture_margin_asymmetric_loss,
                       margin=mixture_loss_config["margin"],
                       energy_masking=energy_masking,
                       target_type=target_type,
                       spec_params=spec_params,
                       mel_scale=mel_scale,
                       mel_params=mel_params)
    else:
        raise ValueError("Invalid mixture loss type: {}".format(mixture_loss_config["name"]))

