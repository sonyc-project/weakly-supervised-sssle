import torch
import torch.nn.functional as F
from functools import partial


def get_mixture_loss_spec_terms(x, labels, masks, energy_mask, energy_masking=None):
    batch_size, n_channel, n_freq, n_time = x.size()
    assert n_channel == 1
    num_labels = labels.size()[-1]

    present_spec = torch.zeros_like(x)
    absent_spec = torch.zeros_like(x)

    for idx in range(num_labels):
        mask = masks[..., idx]
        x_masked = x * mask

        present_spec += x_masked * labels[:, None, None, idx:idx+1]
        absent_spec += x_masked * (1 - labels[:, None, None, idx:idx+1])

    mix_present_spec_diff = x - present_spec

    if energy_masking:
        energy_mask = energy_mask[:, None, None, :]
        mix_present_spec_diff *= energy_mask
        absent_spec *= energy_mask

    mix_present_spec_diff_flat = mix_present_spec_diff.view(batch_size, -1)
    absent_spec_flat = absent_spec.view(batch_size, -1)
    return mix_present_spec_diff_flat, absent_spec_flat


def get_normalization_factor(x, energy_mask, energy_masking=None, squeeze=False):
    batch_size, n_channel, n_freq, n_time = x.size()
    if energy_masking is None:
        return torch.tensor(n_freq * n_time, dtype=x.dtype, device=x.device)
    else:
        norm_factor = energy_mask.sum(dim=-1, keepdims=True)[:, None, None, :]
        if squeeze:
            norm_factor = norm_factor.squeeze()
        return norm_factor


def mixture_loss(x, labels, masks, energy_mask, energy_masking=None):
    mix_present_spec_diff_flat, absent_spec_flat = get_mixture_loss_spec_terms(x, labels, masks, energy_mask, energy_masking)
    norm_factor = get_normalization_factor(x, energy_mask, energy_masking=energy_masking)

    present_energy_diff = torch.norm(mix_present_spec_diff_flat, p=1, dim=1) / norm_factor
    absent_energy = torch.norm(absent_spec_flat, p=1, dim=1) / norm_factor

    mix_loss = (present_energy_diff + absent_energy).mean()
    return mix_loss


def mixture_margin_loss(x, labels, masks, margin, energy_mask, energy_masking=None):
    mix_present_spec_diff_flat, absent_spec_flat = get_mixture_loss_spec_terms(x, labels, masks, energy_mask, energy_masking)
    norm_factor = get_normalization_factor(x, energy_mask, energy_masking=energy_masking)

    present_loss = F.relu(torch.norm(mix_present_spec_diff_flat, p=1, dim=1) - margin) / norm_factor
    absent_loss = torch.norm(absent_spec_flat, p=1, dim=1) / norm_factor
    mix_loss = (present_loss + absent_loss).mean()
    return mix_loss


def mixture_margin_asymmetric_loss(x, labels, masks, margin, energy_mask, energy_masking=None):
    mix_present_spec_diff_flat, absent_spec_flat = get_mixture_loss_spec_terms(x, labels, masks, energy_mask, energy_masking)
    norm_factor = get_normalization_factor(x, energy_mask, energy_masking=energy_masking)

    present_underest = F.relu(F.relu(mix_present_spec_diff_flat).sum(dim=1) - margin) / norm_factor
    present_overest = F.relu(-mix_present_spec_diff_flat).sum(dim=1) / norm_factor
    present_term = present_underest + present_overest
    absent_energy = torch.norm(absent_spec_flat, p=1, dim=1) / norm_factor

    mix_loss = (present_term + absent_energy).mean()
    return mix_loss


def get_mixture_loss_function(train_config):
    mixture_loss_config = train_config["losses"]["mixture"]
    energy_masking = mixture_loss_config.get("energy_masking", False)

    if mixture_loss_config["name"] == "mixture_loss":
        return partial(mixture_loss,
                       energy_masking=energy_masking)
    elif mixture_loss_config["name"] == "mixture_margin_loss":
        return partial(mixture_margin_loss, margin=mixture_loss_config["margin"],
                       energy_masking=energy_masking)
    elif mixture_loss_config["name"] == "mixture_margin_asymmetric_loss":
        return partial(mixture_margin_asymmetric_loss,
                       margin=mixture_loss_config["margin"],
                       energy_masking=energy_masking)
    else:
        raise ValueError("Invalid mixture loss type: {}".format(mixture_loss_config["name"]))

