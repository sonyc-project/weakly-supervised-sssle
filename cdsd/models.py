import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from data import get_data_transforms
from utils import same_pad, conv2d_output_shape
from copy import deepcopy


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

        x = self._forward(x)
        return x


def chainer_init_blstm(blstm):
    """Initialize BLSTM weights a la Chainer defaults"""
    for name, param in blstm.named_parameters():
        if 'weight' in name:
            nn.init.kaiming_normal_(param)
        elif 'bias' in name:
            # Bias structured as [b_ig | b_fg | b_gg | b_og]
            # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745/2
            # https://www.gregcondit.com/articles/lstm-ref-card
            size = param.shape[0]
            forget_start = size // 4
            forget_end = size // 2
            # Init non-forget weights
            nn.init.constant_(param[:forget_start], 0.0)
            nn.init.constant_(param[forget_end:], 0.0)
            # Init forget weights
            nn.init.constant_(param[forget_start:forget_end], 1.0)


def chainer_init_fc(fc):
    """Initialize FC weights a la Chainer defaults"""
    for name, param in fc.named_parameters():
        if 'weight' in name:
            nn.init.kaiming_normal_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)


def chainer_init_conv(conv):
    """Initialize convolution weights a la Chainer defaults"""
    for name, param in conv.named_parameters():
        if 'weight' in name:
            nn.init.kaiming_normal_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)


class BLSTMSpectrogramSeparator(Separator):
    def __init__(self, n_bins, n_classes, n_layers=3, hidden_dim=600, bias=False, transform=None):
        super(BLSTMSpectrogramSeparator, self).__init__(n_classes, transform=transform)
        self.blstm = nn.LSTM(input_size=n_bins,
                             hidden_size=hidden_dim,
                             num_layers=n_layers,
                             batch_first=True,
                             bias=bias,
                             bidirectional=True)
        chainer_init_blstm(self.blstm)
        self.blstm.flatten_parameters()

        self.fc = nn.Linear(2*hidden_dim, n_bins * n_classes, bias=bias)
        chainer_init_fc(self.fc)
        self.n_bins = n_bins

    def _forward(self, x):
        # Remove channel dimension
        x = x.squeeze(dim=1)

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

        # Add channel dimension back in
        x = x[:, None, ...]
        return x


class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, nonlinearity='relu', leakiness=0.2):
        super().__init__()

        if nonlinearity == 'relu':
            nl_layer = nn.ReLU()
        elif nonlinearity == 'leaky_relu':
            nl_layer = nn.LeakyReLU(leakiness)
        else:
            raise ValueError('Invalid nonlinearity: {}'.format(nonlinearity))

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(out_channels),
            nl_layer,
        )

    def forward(self, x):
        return self.conv(x)


class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, skip=True,
                 even_dims=(False, False), nonlinearity='relu',
                 dropout=False, p_dropout=0.5):
        super().__init__()
        output_padding = tuple(1 if x else 0 for x in even_dims)

        if nonlinearity == 'relu':
            nl_layer = nn.ReLU()
        elif nonlinearity == 'sigmoid':
            nl_layer = nn.Sigmoid()
        else:
            raise ValueError('Invalid nonlinearity: {}'.format(nonlinearity))

        modules = [
            nn.ConvTranspose2d(in_channels * 2 if skip else in_channels,
                               out_channels,
                               kernel_size=5,
                               stride=2,
                               padding=2,
                               output_padding=output_padding),
            nn.BatchNorm2d(out_channels),
            nl_layer,
        ]

        if dropout:
            modules.append(nn.Dropout2d(p=p_dropout))

        self.deconv = nn.Sequential(*modules)
        self.skip = skip

    def forward(self, x1, x2=None):
        if x2 is not None and self.skip:
            x1 = torch.cat([x2, x1], dim=1)
        return self.deconv(x1)


class UNetSpectrogramSeparator(Separator):
    def __init__(self, n_bins, n_frames, n_classes, n_blocks=5, init_channels=16,
                 transform=None, janssen_variant=False, **kwargs):
        super(UNetSpectrogramSeparator, self).__init__(n_classes,
                                                       transform=transform)
        self.n_channels = 1
        self.n_classes = n_classes
        self.n_blocks = n_blocks

        num_channels = init_channels
        output_shape = (n_bins, n_frames)
        self.block_data_shapes = [output_shape]

        # Janssen et al. variant uses Leaky ReLU
        if janssen_variant:
            self.inc = UNetDown(self.n_channels, num_channels,
                                nonlinearity='leaky_relu', leakiness=0.2)
        else:
            self.inc = UNetDown(self.n_channels, num_channels,
                                nonlinearity='relu')
        output_shape = conv2d_output_shape(output_shape,
                                           conv_layer=self.inc.conv[0])
        self.block_data_shapes.append(output_shape)

        # Construct down blocks
        self.down_layers = []
        for block_idx in range(n_blocks):
            # Janssen et al. variant uses Leaky ReLU
            if janssen_variant:
                layer = UNetDown(num_channels, num_channels * 2,
                                 nonlinearity='leaky_relu', leakiness=0.2)
            else:
                layer = UNetDown(num_channels, num_channels * 2,
                                 nonlinearity='relu')
            self.down_layers.append(layer)
            # Register layer
            self.add_module("down" + str(block_idx + 1), layer)
            # Get output shape for adding padding to up blocks
            output_shape = conv2d_output_shape(output_shape,
                                               conv_layer=layer.conv[0])
            self.block_data_shapes.append(output_shape)
            # Update number of channels
            num_channels *= 2

        # Construct up blocks
        self.up_layers = []
        for block_idx in range(n_blocks):
            # First up block does not have a skip connection
            skip = (block_idx != 0)
            # Figure out if we need to pad to get the right output shape
            input_shape = self.block_data_shapes[-(2 + block_idx)]
            even_dims = (input_shape[0] % 2 == 0, input_shape[1] % 2 == 0)
            # Janssen et al. variant uses drop out in the first half of
            # up blocks
            if janssen_variant and (block_idx < ((n_blocks + 1) // 2)):
                layer = UNetUp(num_channels, num_channels // 2,
                               skip=skip, even_dims=even_dims,
                               dropout=True, p_dropout=0.5)
            else:
                layer = UNetUp(num_channels, num_channels // 2,
                               skip=skip, even_dims=even_dims, dropout=False)
            self.up_layers.append(layer)
            # Register layer
            self.add_module("up" + str(block_idx + 1), layer)
            # Update number of channels
            num_channels //= 2

        # Sanity check
        assert num_channels == init_channels
        # Set up output layer

        input_shape = self.block_data_shapes[-(2 + n_blocks)]
        even_dims = (input_shape[0] % 2 == 0, input_shape[1] % 2 == 0)
        self.outc = UNetUp(init_channels,
                           n_classes,
                           even_dims=even_dims,
                           nonlinearity='sigmoid')

    def _forward(self, x):
        x_down_list = []
        x = self.inc(x)
        x_down_list.append(x)
        for down_layer in self.down_layers:
            x = down_layer(x)
            x_down_list.append(x)

        for block_idx, up_layer in enumerate(self.up_layers):
            if block_idx > 0:
                x_skip = x_down_list[-(1 + block_idx)]
            else:
                x_skip = None
            x = up_layer(x, x_skip)

        mask = self.outc(x, x_down_list[0])
        return mask[..., None].transpose(1, -1).contiguous()


class Classifier(nn.Module):
    def __init__(self, n_classes, pooling='max', transform=None):
        super(Classifier, self).__init__()
        self.n_classes = n_classes
        self.pooling = pooling
        self.transform = transform

    # JTC: Classifiers are expected to output some kind of frame-wise estimate
    def forward_frame(self, x):
        raise NotImplementedError()

    # JTC: Framewise estimates are pooled in some fashion, e.g. max pooling
    def forward(self, x):
        if self.transform is not None:
            x = self.transform(x)
        x = self.forward_frame(x)

        if self.pooling == 'max':
            # Take the max over the time dimension
            x, _ = x.max(dim=1)
        elif self.pooling is not None:
            raise ValueError('Invalid pooling type: {}'.format(self.pooling))
        # TODO: Implement Autopool

        # If self.pooling is None, do not pool

        return x

    def get_num_frames(self, input_frames):
        return input_frames


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
        chainer_init_blstm(self.blstm)
        self.blstm.flatten_parameters()

        # PyTorch automatically takes care of time-distributing for linear layers
        self.fc = nn.Linear(2*hidden_dim, self.n_classes, bias=bias)
        chainer_init_fc(self.fc)

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
                 pooling='max', num_input_channels=1, conv_kernel_size=(5,5),
                 transform=None):
        super(CRNNSpectrogramClassifier, self).__init__(n_classes,
                                                        pooling=pooling,
                                                        transform=transform)

        self.n_bins = n_bins
        self.conv_kernel_size = conv_kernel_size
        self.conv_padding = same_pad(conv_kernel_size)

        self.conv1_cout = 32
        self.conv1 = nn.Conv2d(in_channels=num_input_channels,
                               out_channels=self.conv1_cout,
                               kernel_size=conv_kernel_size,
                               padding=self.conv_padding,
                               bias=bias)
        chainer_init_conv(self.conv1)
        self.conv1_bin_out = conv2d_output_shape(n_bins,
                                            kernel_size=conv_kernel_size,
                                            padding=self.conv_padding)[0]
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
        chainer_init_conv(self.conv2)
        self.conv2_bin_out = conv2d_output_shape(self.pool1_bin_out,
                                            kernel_size=conv_kernel_size,
                                            padding=self.conv_padding)[0]
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
        chainer_init_conv(self.conv3)
        self.conv3_bin_out = conv2d_output_shape(self.pool2_bin_out,
                                            kernel_size=conv_kernel_size,
                                            padding=self.conv_padding)[0]
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
        chainer_init_blstm(self.blstm)
        self.blstm.flatten_parameters()

        # PyTorch automatically takes care of time-distributing for linear layers
        self.fc = nn.Linear(2*blstm_hidden_dim, self.n_classes, bias=bias)
        chainer_init_fc(self.fc)

    def get_num_frames(self, num_input_frames):
        conv1_frame_out = conv2d_output_shape(num_input_frames,
                                              kernel_size=self.conv_kernel_size,
                                              padding=self.conv_padding)[1]

        pool1_frame_out = conv2d_output_shape(conv1_frame_out,
                                              kernel_size=self.pool1_size,
                                              stride=self.pool1_size)[1]

        conv2_frame_out = conv2d_output_shape(pool1_frame_out,
                                              kernel_size=self.conv_kernel_size,
                                              padding=self.conv_padding)[1]

        pool2_frame_out = conv2d_output_shape(conv2_frame_out,
                                              kernel_size=self.pool2_size,
                                              stride=self.pool2_size)[1]

        conv3_frame_out = conv2d_output_shape(pool2_frame_out,
                                              kernel_size=self.conv_kernel_size,
                                              padding=self.conv_padding)[1]
        pool3_frame_out = conv2d_output_shape(conv3_frame_out,
                                              kernel_size=self.pool3_size,
                                              stride=self.pool3_size)[1]

        return pool3_frame_out

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


def construct_separator(train_config, dataset, weights_path=None, require_init=False, trainable=True, device=None, checkpoint='best'):
    ## Build separator
    separator_config = train_config["separator"]

    # Set up input transformations for separator
    separator_input_transform = get_data_transforms(separator_config)

    num_classes = dataset.num_labels
    separate_background = train_config["training"].get("separate_background", False)
    if separate_background:
        num_classes += 1

    input_shape = dataset.get_input_shape()
    separator_params = deepcopy(separator_config["parameters"])

    # Construct separator
    if separator_config["model"] == "BLSTMSpectrogramSeparator":
        # For backwards compatibility
        separator_params.pop("n_bins", None)
        separator_params.pop("n_frames", None)
        _, n_bins, n_frames = input_shape
        separator = BLSTMSpectrogramSeparator(n_bins=n_bins,
                                              n_classes=num_classes,
                                              transform=separator_input_transform,
                                              **separator_params)
    elif separator_config["model"] == "UNetSpectrogramSeparator":
        # For backwards compatibility
        separator_params.pop("n_bins", None)
        separator_params.pop("n_frames", None)
        _, n_bins, n_frames = input_shape
        separator = UNetSpectrogramSeparator(n_bins=n_bins,
                                             n_frames=n_frames,
                                             n_classes=num_classes,
                                             transform=separator_input_transform,
                                             **separator_params)
    else:
        raise ValueError("Invalid separator model type: {}".format(separator_config["model"]))

    ## Load pretrained model weights for separator if specified
    if not weights_path:
        if checkpoint == 'best':
            weights_path = separator_config.get("best_path") \
                           or separator_config.get("pretrained_path")
        else:
            weights_path = separator_config.get(checkpoint + "_path")

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


def construct_classifier(train_config, dataset, label_mode, weights_path=None, require_init=False, trainable=True, device=None, checkpoint='best'):
    ## Build classifier
    classifier_config = train_config["classifier"]

    # Set up input transformations for separator
    classifier_input_transform = get_data_transforms(classifier_config)

    pooling = classifier_config["parameters"].get("pooling", "max")
    if label_mode == "clip":
        assert pooling is not None
    elif label_mode == "frame":
        assert pooling is None

    input_shape = dataset.get_input_shape()
    classifier_params = deepcopy(classifier_config["parameters"])

    # Construct classifier
    if classifier_config["model"] == "BLSTMSpectrogramClassifier":
        classifier_params.pop("n_bins", None)
        _, n_bins, _ = input_shape
        classifier = BLSTMSpectrogramClassifier(n_bins=n_bins,
                                                n_classes=dataset.num_labels,
                                                transform=classifier_input_transform,
                                                **classifier_params)
    elif classifier_config["model"] == "CRNNSpectrogramClassifier":
        classifier_params.pop("n_bins", None)
        _, n_bins, _ = input_shape
        classifier = CRNNSpectrogramClassifier(n_bins=n_bins,
                                               n_classes=dataset.num_labels,
                                               transform=classifier_input_transform,
                                               **classifier_params)
    else:
        raise ValueError("Invalid classifier model type: {}".format(classifier_config["model"]))

    # Load pretrained model weights for classifier if specified
    if not weights_path:
        if checkpoint == 'best':
            weights_path = classifier_config.get("best_path") \
                           or classifier_config.get("pretrained_path")
        else:
            weights_path = classifier_config.get(checkpoint + "_path")

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
