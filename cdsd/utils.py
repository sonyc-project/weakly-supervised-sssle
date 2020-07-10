import math
import os
import sys
from contextlib import contextmanager
from torch import hann_window, sqrt, ones
from torch.optim import Adam, SGD

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

@contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

def num2tuple(num):
    return num if isinstance(num, tuple) else (num, num)


def conv2d_output_shape(h_w, kernel_size=1, stride=1, padding=0, dilation=1):
    # https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/7
    h_w, kernel_size, stride, padding, dilation = num2tuple(h_w), \
                                              num2tuple(kernel_size), num2tuple(
        stride), num2tuple(padding), num2tuple(dilation)
    padding = num2tuple(padding[0]), num2tuple(padding[1])

    h = math.floor(
        (h_w[0] + sum(padding[0]) - dilation[0] * (kernel_size[0] - 1) - 1) /
        stride[0] + 1)
    w = math.floor(
        (h_w[1] + sum(padding[1]) - dilation[1] * (kernel_size[1] - 1) - 1) /
        stride[1] + 1)

    return h, w


def same_pad(kernel_size):
    return tuple((n-1)//2 for n in num2tuple(kernel_size))


def sqrt_hann_window(*args, **kwargs):
    return sqrt(hann_window(*args, **kwargs))


def get_torch_window_fn(name):
    if name == "hann_window":
        return hann_window
    elif name == "sqrt_hann_window":
        return sqrt_hann_window
    elif name == 'rectangular':
        return ones
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

