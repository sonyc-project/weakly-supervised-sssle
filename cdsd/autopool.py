import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data


class AutoPool1D(nn.Module):
    def __init__(self, dim=0):
        super(AutoPool1D, self).__init__()
        # By default, initialize to all zeros
        self.kernel = torch.nn.Parameter(data=torch.Tensor(1, 1, 1, 1),
                                         requires_grad=True)

        self.dim = dim

        # Initialize as all zeros
        nn.init.constant(self.kernel.data, 0.0)

    def forward(self, x):
        scaled = self.kernel * x
        _, max_val = scaled.max(dim=self.dim + 1, keepdim=True)
        softmax = torch.exp(scaled - max_val)
        weights = softmax / softmax.sum(dim=self.dim + 1, keepdim=True)
        return torch.sum(x * weights, dim=self.dim + 1, keepdim=False)

    # For constraints, see Keras implementation: https://github.com/keras-team/keras/blob/master/keras/constraints.py#L61
    # JTC: I suppose the idea would be to return loss terms that can be easily
    #      added to the total loss. Probably good to see how this is usually handled
    def nonnegative_constraint(self):
        pass
        # From Keras: return w * K.cast(K.greater_equal(w, 0.), K.floatx())