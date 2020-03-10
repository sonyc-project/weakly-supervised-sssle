import torch
import torch.nn.functional as F
from functools import partial


def get_energy_terms(x, labels, masks, energy_masking=None):
    batch_size, n_channel, n_freq, n_time = x.size()
    num_labels = labels.size()[-1]

    if energy_masking is None:
        mix_energy = x.view(batch_size, -1)
    else:
        frame_energy = x.sum(dim=2, keepdim=True)
        threshold = frame_energy.max(dim=3, keepdim=True)[0] * 0.01
        energy_mask = (frame_energy >= threshold).float()
        mix_energy = (x * energy_mask).view(batch_size, -1)

    present_energy = torch.zeros_like(mix_energy)
    absent_energy = torch.zeros_like(mix_energy)

    for idx in range(num_labels):
        mask = masks[..., idx]
        x_masked = x * mask

        if energy_masking is not None:
            x_masked = x_masked * energy_mask

        present_energy += x_masked.view(batch_size, -1) * labels[:, idx:idx+1]
        absent_energy += x_masked.view(batch_size, -1) * (1 - labels[:, idx:idx+1])

    return mix_energy, present_energy, absent_energy


def mixture_loss(x, labels, masks, energy_masking=None):
    mix_energy, present_energy, absent_energy = get_energy_terms(x, labels, masks, energy_masking=energy_masking)
    present_loss = torch.norm(mix_energy - present_energy, p=1, dim=1)
    absent_loss = torch.norm(absent_energy, p=1, dim=1)
    mix_loss = (present_loss + absent_loss).mean()
    return mix_loss


def mixture_margin_loss(x, labels, masks, margin, energy_masking=None):
    mix_energy, present_energy, absent_energy = get_energy_terms(x, labels, masks, energy_masking=energy_masking)
    present_loss = F.relu(torch.norm(mix_energy - present_energy, p=1, dim=1) - margin)
    absent_loss = torch.norm(absent_energy, p=1, dim=1)
    mix_loss = (present_loss + absent_loss).mean()
    return mix_loss


def mixture_margin_asymmetric_loss(x, labels, masks, margin, energy_masking=None):
    mix_energy, present_energy, absent_energy = get_energy_terms(x, labels, masks, energy_masking=energy_masking)
    present_underest_loss = F.relu(F.relu(mix_energy - present_energy).sum(dim=1) - margin)
    present_overest_loss = F.relu(present_energy - mix_energy).sum(dim=1)
    present_loss = present_underest_loss + present_overest_loss
    absent_loss = torch.norm(absent_energy, p=1, dim=1)

    mix_loss = (present_loss + absent_loss).mean()
    return mix_loss


def get_mixture_loss_function(train_config):
    mixture_loss_config = train_config["losses"]["mixture"]
    if mixture_loss_config["name"] == "mixture_loss":
        return partial(mixture_loss,
                       energy_masking=mixture_loss_config.get("energy_masking", False))
    elif mixture_loss_config["name"] == "mixture_margin_loss":
        return partial(mixture_margin_loss, margin=mixture_loss_config["margin"],
                       energy_masking=mixture_loss_config.get("energy_masking", False))
    elif mixture_loss_config["name"] == "mixture_margin_asymmetric_loss":
        return partial(mixture_margin_asymmetric_loss,
                       margin=mixture_loss_config["margin"],
                       energy_masking=mixture_loss_config.get("energy_masking", False))
    else:
        raise ValueError("Invalid mixture loss type: {}".format(mixture_loss_config["name"]))
