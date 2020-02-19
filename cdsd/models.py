import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


class Separator(nn.Module):
    pass


class BLSTMSpectrogramSeparator(Separator):
    def __init__(self, n_bins, n_layers=3, hidden_dim=600, bias=False):
        super(BLSTMSpectrogramSeparator, self).__init__()
        self.blstm = nn.LSTM(input_size=n_bins,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bias=bias,
                              bidirectional=True)

        self.fc = nn.Linear(2*hidden_dim, n_bins, bias=bias)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        # Reshape since PyTorch convention is for time to be the last dimension,
        # but we need to operate on the channel dimension
        x = x.transpose(2, 1)
        x, _ = self.blstm(x)
        # JTC: IIUC, then the nonlinearity is baked into the LSTM implementation
        # PyTorch automatically takes care of time-distributing for linear layers
        x = torch.sigmoid(self.fc(x))
        return x


class Classifier(nn.Module):
    def __init__(self, n_classes, pooling='max'):
        super(Classifier, self).__init__()
        self.n_classes = n_classes
        self.pooling = pooling

    # JTC: Classifiers are expected to output some kind of frame-wise estimate
    def forward_frame(self, x):
        raise NotImplementedError()

    # JTC: Framewise estiamtes are pooled in some fashion, e.g. max pooling
    def forward(self, x):
        x = self.forward_frame(x)
        if self.pooling == 'max':
            # Take the max over the time dimension
            _, x = x.max(dim=1)
        else:
            raise ValueError('Invalid pooling type: {}'.format(self.pooling))
        return x


class BLSTMClassifier(Classifier):
    def __init__(self, n_bins, n_classes, n_layers=2, hidden_dim=100, bias=False,
                 pooling='max'):
        super(BLSTMClassifier, self).__init__(n_classes, pooling)
        self.blstm = nn.LSTM(input_size=n_bins,
                             hidden_size=hidden_dim,
                             num_layers=n_layers,
                             bias=bias,
                             bidirectional=True)

        # PyTorch automatically takes care of time-distributing for linear layers
        self.fc = nn.Linear(2*hidden_dim, self.n_classes, bias=bias)

        # TODO: Implement Autopool

    def forward_frame(self, x):
        x = x.transpose(2, 1)
        x, _ = self.blstm(x)
        x = torch.sigmoid(self.fc(x))
        return x
