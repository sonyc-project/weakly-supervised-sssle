from torch import hann_window, sqrt
from torch.optim import Adam, SGD


def sqrt_hann_window(*args, **kwargs):
    return sqrt(hann_window(*args, **kwargs))


def get_torch_window_fn(name):
    if name == "hann_window":
        return hann_window
    elif name == "sqrt_hann_window":
        return sqrt_hann_window
    else:
        raise ValueError('Invalid window type: {}'.format(name))


def get_optimizer(parameters, train_config):
    opt_config = train_config["training"]["optimizer"]
    opt_name = opt_config["name"]
    opt_params = opt_config["parameters"]

    if opt_name == "Adam":
        return Adam(parameters, **opt_params)
    elif opt_name == "SGD":
        return SGD(parameters, **opt_params)
    else:
        raise ValueError("Invalid optimizer: {}".format(opt_name))

