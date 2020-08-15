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


def transform_spec_to_target(mix_present_spec_diff, present_spec, absent_spec, x, target_type, spec_params=None, mel_scale=False, mel_params=None):
    batch_size, n_channel, n_freq, n_time = x.size()
    if target_type == "timefreq":
        mix_present_spec_diff = mix_present_spec_diff.reshape(batch_size, -1)
        absent_spec = absent_spec.reshape(batch_size, -1)
    elif target_type == "spectrum":
        # Average over time and channels
        mix_present_spec_diff = mix_present_spec_diff.mean(dim=-1).mean(dim=1)
        absent_spec = absent_spec.mean(dim=-1).mean(dim=1)
    elif target_type == "dbfs":
        x_dbfs = compute_dbfs_spec(x, SAMPLE_RATE, spec_params=spec_params, mel_scale=mel_scale, mel_params=mel_params, device=x.device)
        present_dbfs = compute_dbfs_spec(present_spec, SAMPLE_RATE, spec_params=spec_params, mel_scale=mel_scale, mel_params=mel_params, device=x.device)
        mix_present_spec_diff = (x_dbfs - present_dbfs).unsqueeze(-1)
        absent_spec = compute_dbfs_spec(absent_spec, SAMPLE_RATE, spec_params=spec_params, mel_scale=mel_scale, mel_params=mel_params, device=x.device).unsqueeze(-1)
    elif target_type == "energy":
        # Average over time, freq, and channel dimensions
        mix_present_spec_diff = mix_present_spec_diff.mean(dim=-1).mean(dim=-1).mean(dim=-1, keepdim=True)
        absent_spec = absent_spec.mean(dim=-1).mean(dim=-1).mean(dim=-1, keepdim=True)
    else:
        raise ValueError("Invalid target type: {}".format(target_type))

    return mix_present_spec_diff, absent_spec


def mixture_loss(x, labels, masks, energy_mask, energy_masking=False, target_type="timefreq", spec_params=None, mel_scale=False, mel_params=None):
    mix_present_spec_diff, present_spec, absent_spec, x = get_mixture_loss_spec_terms(x, labels, masks, energy_mask, energy_masking, flatten=(target_type == "timefreq"), mel_scale=mel_scale, mel_params=mel_params)
    mix_present_spec_diff, absent_spec = transform_spec_to_target(mix_present_spec_diff, present_spec, absent_spec, x, target_type, spec_params=spec_params, mel_scale=mel_scale, mel_params=mel_params)
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
    mix_present_spec_diff, absent_spec = transform_spec_to_target(mix_present_spec_diff, present_spec, absent_spec, x, target_type, spec_params=spec_params, mel_scale=mel_scale, mel_params=mel_params)
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
    mix_present_spec_diff, absent_spec = transform_spec_to_target(mix_present_spec_diff, present_spec, absent_spec, x, target_type, spec_params=spec_params, mel_scale=mel_scale, mel_params=mel_params)
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


def create_average_of_losses(loss_func_list, loss_weight_list=None):
    num_losses = len(loss_func_list)

    def f(*args, **kwargs):
        total_loss = None
        for loss_idx, loss_func in enumerate(loss_func_list):
            loss = loss_func(*args, **kwargs)
            if loss_weight_list is not None:
                loss *= loss_weight_list[loss_idx]

            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss
        return total_loss / num_losses
    return f


def get_mixture_loss_function(train_config):
    mixture_loss_config_list = train_config["losses"]["mixture"]
    spec_params = get_spec_params(train_config)

    if type(mixture_loss_config_list) == dict:
        mixture_loss_config_list = [mixture_loss_config_list]
    else:
        assert type(mixture_loss_config_list) == list

    loss_func_list = []
    loss_weight_list = []
    for mixture_loss_config in mixture_loss_config_list:
        loss_name = mixture_loss_config["name"]
        energy_masking = mixture_loss_config.get("energy_masking", False)
        target_type = mixture_loss_config.get("target_type", "timefreq")
        mel_scale = mixture_loss_config.get("mel_scale", False)
        mel_params = mixture_loss_config.get("mel_params", None)
        weight = mixture_loss_config["weight"]

        if loss_name == "mixture_loss":
            loss_func = partial(mixture_loss,
                                energy_masking=energy_masking,
                                target_type=target_type,
                                spec_params=spec_params,
                                mel_scale=mel_scale,
                                mel_params=mel_params)
        elif loss_name == "mixture_margin_loss":
            loss_func = partial(mixture_margin_loss, margin=mixture_loss_config["margin"],
                                energy_masking=energy_masking,
                                target_type=target_type,
                                spec_params=spec_params,
                                mel_scale=mel_scale,
                                mel_params=mel_params)
        elif loss_name == "mixture_margin_asymmetric_loss":
            loss_func = partial(mixture_margin_asymmetric_loss,
                                margin=mixture_loss_config["margin"],
                                energy_masking=energy_masking,
                                target_type=target_type,
                                spec_params=spec_params,
                                mel_scale=mel_scale,
                                mel_params=mel_params)
        else:
            raise ValueError("Invalid mixture loss type: {}".format(mixture_loss_config["name"]))

        loss_func_list.append(loss_func)
        loss_weight_list.append(weight)

    if len(loss_func_list) == 1:
        return loss_func_list[0]
    else:
        return create_average_of_losses(loss_func_list, loss_weight_list=loss_weight_list)


def separation_loss(src_spec, x_masked, weight, energy_mask, mel_tf=None, energy_masking=False, target_type="timefreq", spec_params=None, mel_scale=False, mel_params=None):
    # Optionally apply mel scale
    if mel_scale:
        assert mel_tf is not None
        src_spec = mel_tf(src_spec)
        x_masked = mel_tf(x_masked)

    src_spec_diff = (src_spec - x_masked) * weight
    if energy_masking:
        src_spec_diff = src_spec_diff * energy_mask[:, None, None, :]

    if target_type == "timefreq":
        src_spec_diff = src_spec_diff.reshape(src_spec.size(0), -1)
    elif target_type == "spectrum":
        # Average over time and channels
        src_spec_diff = src_spec_diff.mean(dim=-1).mean(dim=1)
    elif target_type == "energy":
        src_spec_diff = src_spec_diff.mean(dim=-1).mean(dim=-1).mean(dim=-1, keepdim=True)
    elif target_type == "dbfs":
        src_spec_dbfs = compute_dbfs_spec(src_spec, SAMPLE_RATE, spec_params=spec_params, mel_scale=mel_scale, mel_params=mel_params, device=src_spec.device)
        x_masked_dbfs = compute_dbfs_spec(x_masked, SAMPLE_RATE, spec_params=spec_params, mel_scale=mel_scale, mel_params=mel_params, device=src_spec.device)
        src_spec_diff = (src_spec_dbfs - x_masked_dbfs).unsqueeze(-1)
        del src_spec_dbfs, x_masked_dbfs
    else:
        raise ValueError("Invalid target type: {}".format(target_type))

    norm_factor = get_normalization_factor(src_spec, energy_masking=energy_masking, target_type=target_type)
    src_loss = torch.norm(src_spec_diff, p=1, dim=1) / norm_factor
    src_loss = src_loss.mean()
    return src_loss


def get_separation_loss_function(train_config, device=None):
    separation_loss_config_list = train_config["losses"]["separation"]
    spec_params = get_spec_params(train_config)

    if type(separation_loss_config_list) == dict:
        separation_loss_config_list = [separation_loss_config_list]
    else:
        assert type(separation_loss_config_list) == list

    loss_func_list = []
    for separation_loss_config in separation_loss_config_list:
        energy_masking = separation_loss_config.get("energy_masking", False)
        target_type = separation_loss_config.get("target_type", "timefreq")
        mel_scale = separation_loss_config.get("mel_scale", False)
        mel_params = separation_loss_config.get("mel_params", None)
        if mel_params is not None:
            mel_tf = MelScale(sample_rate=SAMPLE_RATE, **mel_params)
            if device is not None:
                mel_tf = mel_tf.to(device)

        loss_func = partial(separation_loss,
                            mel_tf=mel_tf,
                            energy_masking=energy_masking,
                            target_type=target_type,
                            spec_params=spec_params,
                            mel_scale=mel_scale,
                            mel_params=mel_params)

        loss_func_list.append(loss_func)

    if len(loss_func_list) == 1:
        return loss_func_list[0]
    else:
        return create_average_of_losses(loss_func_list)

