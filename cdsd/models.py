import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from data import get_data_transforms
from utils import same_pad, conv2d_output_shape


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

        # Remove channel dimension
        x = x.squeeze(dim=1)
        x = self._forward(x)
        # Add channel dimension back in
        x = x[:, None, ...]
        return x


class BLSTMSpectrogramSeparator(Separator):
    def __init__(self, n_bins, n_classes, n_layers=3, hidden_dim=600, bias=False, transform=None):
        super(BLSTMSpectrogramSeparator, self).__init__(n_classes, transform=transform)
        self.blstm = nn.LSTM(input_size=n_bins,
                             hidden_size=hidden_dim,
                             num_layers=n_layers,
                             batch_first=True,
                             bias=bias,
                             bidirectional=True)
        self.blstm.flatten_parameters()

        self.fc = nn.Linear(2*hidden_dim, n_bins * n_classes, bias=bias)
        self.n_bins = n_bins

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def _forward(self, x):
        batch_size = len(x)
        # Reshape since PyTorch convention is for time to be the last dimension,
        # but we need to operate on the channel dimension
        x = x.transpose(2, 1)
        self.blstm.flatten_parameters()
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
        # TODO: Implement Autopool
        return x


class BLSTMSpectrogramClassifier(Classifier):
    def __init__(self, n_bins, n_classes, n_layers=2, hidden_dim=100, bias=False,
                 pooling='max', transform=None):
        super(BLSTMSpectrogramClassifier, self).__init__(n_classes,
                                                         pooling=pooling,
                                                         transform=transform)
        self.blstm = nn.LSTM(input_size=n_bins,
                             hidden_size=hidden_dim,
                             num_layers=n_layers,
                             bias=bias,
                             batch_first=True,
                             bidirectional=True)
        self.blstm.flatten_parameters()

        # PyTorch automatically takes care of time-distributing for linear layers
        self.fc = nn.Linear(2*hidden_dim, self.n_classes, bias=bias)

    def forward_frame(self, x):
        # Remove channel dimension
        x = x.squeeze(dim=1)
        # Reshape since PyTorch convention is for time to be the last dimension,
        # but we need to operate on the channel dimension
        x = x.transpose(2, 1)
        self.blstm.flatten_parameters()
        x, _ = self.blstm(x)
        x = torch.sigmoid(self.fc(x))
        return x


class CRNNSpectrogramClassifier(Classifier):
    def __init__(self, n_bins, n_classes, blstm_hidden_dim=100, bias=False,
                 pooling='max', num_input_channels=1, conv_kernel_size=(5,5), transform=None):
        super(CRNNSpectrogramClassifier, self).__init__(n_classes,
                                                        pooling=pooling,
                                                        transform=transform)

        conv_padding = same_pad(conv_kernel_size)

        self.conv1_cout = 32
        self.conv1 = nn.Conv2d(in_channels=num_input_channels,
                               out_channels=self.conv1_cout,
                               kernel_size=conv_kernel_size,
                               padding=conv_padding,
                               bias=bias)
        self.conv1_bin_out = conv2d_output_shape(n_bins,
                                            kernel_size=conv_kernel_size,
                                            padding=conv_padding)[0]
        # Transpose of Pishdadian et al. since PyTorch convention puts
        # time in last dimension
        self.pool1_size = (2,1)
        self.pool1 = nn.MaxPool2d(self.pool1_size)
        self.pool1_bin_out = conv2d_output_shape(self.conv1_bin_out,
                                            kernel_size=self.pool1_size,
                                            stride=self.pool1_size)[0]

        self.conv2_cout = 64
        self.conv2 = nn.Conv2d(in_channels=self.conv1_cout,
                               out_channels=self.conv2_cout,
                               kernel_size=conv_kernel_size,
                               padding=same_pad(conv_kernel_size),
                               bias=bias)
        self.conv2_bin_out = conv2d_output_shape(self.pool1_bin_out,
                                            kernel_size=conv_kernel_size,
                                            padding=conv_padding)[0]
        self.pool2_size = (2,2)
        self.pool2 = nn.MaxPool2d(self.pool2_size)
        self.pool2_bin_out = conv2d_output_shape(self.conv2_bin_out,
                                            kernel_size=self.pool2_size,
                                            stride=self.pool2_size)[0]

        self.conv3_cout = 128
        self.conv3 = nn.Conv2d(in_channels=self.conv2_cout,
                               out_channels=self.conv3_cout,
                               kernel_size=conv_kernel_size,
                               padding=same_pad(conv_kernel_size),
                               bias=bias)
        self.conv3_bin_out = conv2d_output_shape(self.pool2_bin_out,
                                            kernel_size=conv_kernel_size,
                                            padding=conv_padding)[0]
        self.pool3_size = (2,2)
        self.pool3 = nn.MaxPool2d(self.pool3_size)
        self.pool3_bin_out = conv2d_output_shape(self.conv3_bin_out,
                                            kernel_size=self.pool3_size,
                                            stride=self.pool3_size)[0]

        self.blstm = nn.LSTM(input_size=self.conv3_cout * self.pool3_bin_out,
                             hidden_size=blstm_hidden_dim,
                             num_layers=1,
                             bias=bias,
                             batch_first=True,
                             bidirectional=True)
        self.blstm.flatten_parameters()

        # PyTorch automatically takes care of time-distributing for linear layers
        self.fc = nn.Linear(2*blstm_hidden_dim, self.n_classes, bias=bias)

    def forward_frame(self, x):
        batch_size = x.size()[0]
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)
        # Flatten the channel and frequency dimensions
        x = x.view(batch_size, self.conv3_cout * self.pool3_bin_out, -1)
        # Switch the feature and time dimensions
        x = x.transpose(2, 1)
        self.blstm.flatten_parameters()
        x, _ = self.blstm(x)
        x = torch.sigmoid(self.fc(x))
        return x


def construct_separator(train_config, dataset, weights_path=None, require_init=False, trainable=True, device=None):
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
                   or separator_config.get("best_path") \
                   or separator_config.get("pretrained_path")
    if weights_path:
        weights = torch.load(weights_path, map_location=device)
        try:
            separator.load_state_dict(weights)
        except RuntimeError:
            print("*** It appears separator was saved in a DataParallel parallel wrapper. Retrieving internal model weights. ***")
            weights = {(k[7:] if k.startswith('module.') else k): v
                       for k, v in weights.items()}
            separator.load_state_dict(weights)
        print("Loaded separator weights from {}".format(weights_path))
    elif require_init:
        raise ValueError("Requires separator weights to be specified.")

    # Freeze separator weights if specified
    if not trainable or not separator_config.get("trainable", True):
        for param in separator.parameters():
            param.requires_grad = False

    return separator


def construct_classifier(train_config, dataset, weights_path=None, require_init=False, trainable=True, device=None):
    ## Build classifier
    classifier_config = train_config["classifier"]

    # Set up input transformations for separator
    classifier_input_transform = get_data_transforms(classifier_config)

    # Construct classifier
    if classifier_config["model"] == "BLSTMSpectrogramClassifier":
        classifier = BLSTMSpectrogramClassifier(n_classes=dataset.num_labels,
                                                transform=classifier_input_transform,
                                                **classifier_config["parameters"])
    if classifier_config["model"] == "CRNNSpectrogramClassifier":
        classifier = CRNNSpectrogramClassifier(n_classes=dataset.num_labels,
                                                transform=classifier_input_transform,
                                                **classifier_config["parameters"])
    else:
        raise ValueError("Invalid classifier model type: {}".format(classifier_config["model"]))

    # Load pretrained model weights for classifier if specified
    weights_path = weights_path \
                   or classifier_config.get("best_path") \
                   or classifier_config.get("pretrained_path")
    if weights_path:
        weights = torch.load(weights_path, map_location=device)
        try:
            classifier.load_state_dict(weights)
        except RuntimeError:
            print("*** It appears classifier was saved in a DataParallel parallel wrapper. Retrieving internal model weights. ***")
            weights = {(k[7:] if k.startswith('module.') else k): v
                       for k, v in weights.items()}
            classifier.load_state_dict(weights)
        print("Loaded classifier weights from {}".format(weights_path))
    elif require_init:
        raise ValueError("Requires classifier weights to be specified.")

    # Freeze classifier weights if specified
    if not trainable or not classifier_config.get("trainable", True):
        for param in classifier.parameters():
            param.requires_grad = False

    return classifier
