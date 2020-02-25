import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from data import get_data_transforms


class Separator(nn.Module):
    def __init__(self, n_classes, transform=None):
        super(Separator, self).__init__()
        self.n_classes = n_classes
        self.transform = transform

    def _forward(self, x):
        raise NotImplementedError()

    def forward(self, x):
        if self.transform is not None:
            x = self.transform(x)
        return self._forward(x)


class BLSTMSpectrogramSeparator(Separator):
    def __init__(self, n_bins, n_classes, n_layers=3, hidden_dim=600, bias=False, transform=None):
        super(BLSTMSpectrogramSeparator, self).__init__(n_classes, transform=transform)
        self.blstm = nn.LSTM(input_size=n_bins,
                             hidden_size=hidden_dim,
                             num_layers=n_layers,
                             bias=bias,
                             bidirectional=True)

        self.fc = nn.Linear(2*hidden_dim, n_bins * n_classes, bias=bias)
        self.n_bins = n_bins

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def _forward(self, x):
        batch_size = len(x)
        # Remove channel dimension
        x = x.squeeze(dim=1)
        # Reshape since PyTorch convention is for time to be the last dimension,
        # but we need to operate on the channel dimension
        x = x.transpose(2, 1)
        x, _ = self.blstm(x)
        # JTC: IIUC, then the nonlinearity is baked into the LSTM implementation
        # PyTorch automatically takes care of time-distributing for linear layers
        x = torch.sigmoid(self.fc(x))
        # Reshape to (batch, n_steps, n_bins, n_classes)
        x = x.view(batch_size, -1, self.n_bins, self.n_classes)
        # Swap frequence and time dims to adhere to PyTorch convention
        x = x.transpose(2, 1)

        return x


class Classifier(nn.Module):
    def __init__(self, n_classes, pooling='max', transform=None):
        super(Classifier, self).__init__()
        self.n_classes = n_classes
        self.pooling = pooling
        self.transform = transform

    # JTC: Classifiers are expected to output some kind of frame-wise estimate
    def forward_frame(self, x):
        raise NotImplementedError()

    # JTC: Framewise estiamtes are pooled in some fashion, e.g. max pooling
    def forward(self, x):
        if self.transform is not None:
            x = self.transform(x)
        x = self.forward_frame(x)
        if self.pooling == 'max':
            # Take the max over the time dimension
            x, _ = x.max(dim=1)
        else:
            raise ValueError('Invalid pooling type: {}'.format(self.pooling))
        return x


class BLSTMSpectrogramClassifier(Classifier):
    def __init__(self, n_bins, n_classes, n_layers=2, hidden_dim=100, bias=False,
                 pooling='max', transform=None):
        super(BLSTMSpectrogramClassifier, self).__init__(n_classes, pooling,
                                                         transform=transform)
        self.blstm = nn.LSTM(input_size=n_bins,
                             hidden_size=hidden_dim,
                             num_layers=n_layers,
                             bias=bias,
                             bidirectional=True)

        # PyTorch automatically takes care of time-distributing for linear layers
        self.fc = nn.Linear(2*hidden_dim, self.n_classes, bias=bias)

        # TODO: Implement Autopool

    def forward_frame(self, x):
        # Remove channel dimension
        x = x.squeeze(dim=1)
        # Reshape since PyTorch convention is for time to be the last dimension,
        # but we need to operate on the channel dimension
        x = x.transpose(2, 1)
        x, _ = self.blstm(x)
        x = torch.sigmoid(self.fc(x))
        return x


def construct_separator(train_config, dataset, weights_path=None, require_init=False):
    ## Build separator
    separator_config = train_config["separator"]

    # Set up input transformations for separator
    separator_input_transform = get_data_transforms(separator_config)

    # Construct separator
    if separator_config["model"] == "BLSTMSpectrogramSeparator":
        separator = BLSTMSpectrogramSeparator(n_classes=dataset.num_labels,
                                              transform=separator_input_transform,
                                              **separator_config["parameters"])
    else:
        raise ValueError("Invalid separator model type: {}".format(separator_config["model"]))

    # Load pretrained model weights for separator if specified
    weights_path = weights_path \
                   or separator_config.get("best_checkpoint_path") \
                   or separator_config.get("pretrained_path")
    if weights_path:
        separator.load_state_dict(torch.load(weights_path))
        print("Loaded separator weights from {}".format(weights_path))
    elif require_init:
        raise ValueError("Requires separator weights to be specified.")

    # Freeze classifier weights if specified
    if not separator_config.get("trainable", True):
        for param in separator.parameters():
            param.requires_grad = False

    return separator


def construct_classifier(train_config, dataset, weights_path=None, require_init=False):
    ## Build classifier
    classifier_config = train_config["classifier"]

    # Set up input transformations for separator
    classifier_input_transform = get_data_transforms(classifier_config)

    # Construct classifier
    if classifier_config["model"] == "BLSTMSpectrogramClassifier":
        classifier = BLSTMSpectrogramClassifier(n_classes=dataset.num_labels,
                                                transform=classifier_input_transform,
                                                **classifier_config["parameters"])
    else:
        raise ValueError("Invalid classifier model type: {}".format(classifier_config["model"]))

    # Load pretrained model weights for classifier if specified
    weights_path = weights_path \
                   or classifier_config.get("best_checkpoint_path") \
                   or classifier_config.get("pretrained_path")
    if weights_path:
        classifier.load_state_dict(torch.load(weights_path))
        print("Loaded classifier weights from {}".format(weights_path))
    elif require_init:
        raise ValueError("Requires classifier weights to be specified.")

    # Freeze classifier weights if specified
    if not classifier_config.get("trainable", True):
        for param in classifier.parameters():
            param.requires_grad = False

    return classifier
