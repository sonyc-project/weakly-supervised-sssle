import torch
import torch.nn.functional as F


def get_source_energy(x, labels, masks):
    batch_size = len(x)
    present_energy = torch.zeros_like(x).view(batch_size, -1)
    absent_energy = torch.zeros_like(x).view(batch_size, -1)

    for idx in range(NUM_LABELS):
        mask = masks[..., idx]
        x_masked = x * mask

        present_energy += x_masked.view(batch_size, -1) * labels[:, idx:idx+1]
        absent_energy += x_masked.view(batch_size, -1) * (1 - labels[:, idx:idx+1])

    return present_energy, absent_energy


def mixture_loss(mix_energy, present_energy, absent_energy):
    present_loss = torch.norm(mix_energy - present_energy, p=1, dim=1)
    absent_loss = torch.norm(absent_energy, p=1, dim=1)
    mix_loss = (present_loss + absent_loss).mean()
    return mix_loss


def mixture_margin_loss(mix_energy, present_energy, absent_energy, margin):
    present_loss = F.relu(torch.norm(mix_energy - present_energy, p=1, dim=1) - margin)
    absent_loss = torch.norm(absent_energy, p=1, dim=1)
    mix_loss = (present_loss + absent_loss).mean()
    return mix_loss


def mixture_margin_asymmetric_loss(mix_energy, present_energy, absent_energy, margin):
    present_underest_loss = F.relu(F.relu(mix_energy - present_energy).sum(dim=1) - margin)
    present_overest_loss = F.relu(present_energy - mix_energy).sum(dim=1)
    present_loss = present_underest_loss + present_overest_loss
    absent_loss = torch.norm(absent_energy, p=1, dim=1)

    mix_loss = (present_loss + absent_loss).mean()
    return mix_loss


