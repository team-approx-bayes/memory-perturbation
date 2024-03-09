import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple
from math import ceil

def get_model(model_name, nc, input_size, device, seed):
    torch.manual_seed(seed)
    if model_name == 'large_mlp':
        hidden_sizes = [500, 300]
        net = MLP(input_size, hidden_sizes, nc).to(device)
    elif model_name == 'small_mlp':
        hidden_sizes = [32, 16]
        net = MLP(input_size, hidden_sizes, nc).to(device)
    elif model_name == 'lenet':
        net = LeNet5(output_size=nc).to(device)
    elif model_name == 'cnn_deepobs':
        net = CIFAR10Net().to(device)
    elif model_name == 'resnet20_frn':
        net = make_resnet20_frn_fn(num_classes=nc).to(device)
    else:
        raise NotImplementedError
    return net

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation="tanh", **kwargs):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        if output_size is not None:
            self.output_size = output_size
        else:
            self.output_size = 1

        # Set activation function
        if activation == "relu":
            self.act = torch.nn.ReLU
        elif activation == "tanh":
            self.act = torch.tanh
        elif activation == "sigmoid":
            self.act = torch.sigmoid
        elif activation == "selu":
            self.act = F.elu

        # Define layers
        if len(hidden_sizes) == 0:
            # Linear model
            self.hidden_layers = []
            self.output_layer = nn.Linear(self.input_size, self.output_size)
        else:
            # Neural network
            in_outs = zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)
            self.hidden_layers = nn.ModuleList([nn.Linear(in_size, out_size) for in_size, out_size
                                                in in_outs])
            self.output_layer = nn.Linear(hidden_sizes[-1], self.output_size)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        out = x
        for layer in self.hidden_layers:
            out = self.act(layer(out))
        z = self.output_layer(out)
        return z.flatten() if self.output_size == 1 else z

class LeNet5(nn.Module):
    def __init__(self, output_size=10):
        super().__init__()
        self.output_size = output_size

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.act1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.act2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        self.act3 = nn.Tanh()

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(1 * 1 * 120, 84)
        self.act4 = nn.Tanh()
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        # input 1x28x28, output 6x28x28
        x = self.act1(self.conv1(x))
        # input 6x28x28, output 6x14x14
        x = self.pool1(x)
        # input 6x14x14, output 16x10x10
        x = self.act2(self.conv2(x))
        # input 16x10x10, output 16x5x5
        x = self.pool2(x)
        # input 16x5x5, output 120x1x1
        x = self.act3(self.conv3(x))
        # input 120x1x1, output 84
        x = self.act4(self.fc1(self.flat(x)))
        # input 84, output 10
        x = self.fc2(x)
        return x

def make_resnet20_frn_fn(num_classes, activation=torch.nn.ReLU):
    return make_resnet_fn(
        num_classes, depth=20, normalization_layer=FilterResponseNorm_layer,
        activation=activation)

class make_resnet_fn(nn.Module):
    def __init__(self, num_classes, depth, normalization_layer,
                 width=16, use_bias=True, activation=torch.nn.ReLU(inplace=True)):
        super(make_resnet_fn, self).__init__()
        self.output_size = 10
        self.num_res_blocks = (depth - 2) // 6
        self.normalization_layer = normalization_layer
        self.activation = activation
        self.use_bias = use_bias
        self.width = width
        if (depth - 2) % 6 != 0:
            raise ValueError('depth must be 6n+2 (e.g. 20, 32, 44).')

        # first res_layer
        self.layer1 = resnet_block(normalization_layer=normalization_layer, num_filters=width,
                                   input_size=(3, 32, 32), kernel_size=3, strides=1,
                                   activation=torch.nn.Identity, use_bias=True)
        # stacks
        self.stacks = self._make_res_block()
        # avg pooling
        self.avgpool1 = torch.nn.AvgPool2d(kernel_size=(8, 8), stride=8, padding=0)
        # linear layer
        self.linear1 = nn.Linear(64, num_classes)

    def forward(self, x):
        # first res_layer
        out = self.layer1(x)  # shape out torch.Size([5, 16, 32, 32])
        out = self.stacks(out)
        out = self.avgpool1(out)
        out = torch.flatten(out, start_dim=1)
        logits = self.linear1(out)
        return logits

    def _make_res_block(self):
        layers = list()
        num_filters = self.width
        input_num_filters = num_filters
        for stack in range(3):
            for res_block in range(self.num_res_blocks):
                layers.append(stacked_resnet_block(self.normalization_layer, num_filters, input_num_filters,
                                                   stack, res_block, self.activation, self.use_bias))
                input_num_filters = num_filters
            num_filters *= 2
        return nn.Sequential(*layers)

class stacked_resnet_block(nn.Module):
    def __init__(self, normalization_layer, num_filters, input_num_filters,
                 stack, res_block, activation, use_bias):
        super(stacked_resnet_block, self).__init__()
        self.stack = stack
        self.res_block = res_block
        spatial_out = 32 // (2 ** stack)
        if stack > 0 and res_block == 0:  # first layer but not first stack
            strides = 2  # downsample
        else:
            strides = 1
        spatial_in = spatial_out * strides

        self.res1 = resnet_block(
            normalization_layer=normalization_layer, num_filters=num_filters,
            input_size=(input_num_filters, spatial_in, spatial_in),
            strides=strides, activation=activation, use_bias=use_bias)
        self.res2 = resnet_block(
            normalization_layer=normalization_layer, num_filters=num_filters,
            input_size=(num_filters, spatial_out, spatial_out),
            use_bias=use_bias)
        if stack > 0 and res_block == 0:  # first layer but not first stack
            # linear projection residual shortcut to match changed dims
            self.res3 = resnet_block(
                normalization_layer=normalization_layer,
                num_filters=num_filters,
                input_size=(input_num_filters, spatial_in, spatial_in),
                strides=strides,
                kernel_size=1,
                use_bias=use_bias)

        self.activation1 = activation()

    def forward(self, x):

        y = self.res1(x)
        y = self.res2(y)
        if self.stack > 0 and self.res_block == 0:
            x = self.res3(x)
        out = self.activation1(x + y)
        return out

class resnet_block(nn.Module):
    def __init__(
            self, normalization_layer, input_size, num_filters, kernel_size=3,
            strides=1, activation=torch.nn.Identity, use_bias=True):
        super(resnet_block, self).__init__()
        # input size = C, H, W
        p0, p1 = conv_same_padding(input_size[2], kernel_size, strides)
        # height padding
        p2, p3 = conv_same_padding(input_size[1], kernel_size, strides)
        self.pad1 = torch.nn.ZeroPad2d((p0, p1, p2, p3))
        self.conv1 = torch.nn.Conv2d(
            input_size[0], num_filters, kernel_size=kernel_size,
            stride=strides, padding=0, bias=use_bias)
        self.norm1 = normalization_layer(num_filters)
        self.activation1 = activation()

    def forward(self, x):

        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.activation1(out)

        return out

# cf. https://github.com/pytorch/pytorch/issues/3867#issuecomment-349279036
# should replicate tensorflow "SAME" padding behavior
def conv_same_padding(
        in_size: int, kernel: int, stride: int = 1, dilation: int = 1
) -> Tuple[int, int]:
    effective_filter_size = (kernel - 1) * dilation + 1
    out_size = (in_size + stride - 1) // stride
    padding_needed = max(
        0, (out_size - 1) * stride + effective_filter_size - in_size)
    if padding_needed % 2 == 0:
        padding_left = padding_needed // 2
        padding_right = padding_needed // 2
    else:
        padding_left = (padding_needed - 1) // 2
        padding_right = (padding_needed + 1) // 2
    return padding_left, padding_right

class FilterResponseNorm_layer(nn.Module):
    def __init__(self, num_filters, eps=1e-6):
        super(FilterResponseNorm_layer, self).__init__()
        self.eps = eps
        par_shape = (1, num_filters, 1, 1)  # [1,C,1,1]
        self.tau = torch.nn.Parameter(torch.zeros(par_shape))
        self.beta = torch.nn.Parameter(
            torch.zeros(par_shape))
        self.gamma = torch.nn.Parameter(
            torch.ones(par_shape))

    def forward(self, x):
        nu2 = torch.mean(torch.square(x), dim=[2, 3], keepdim=True)
        x = x * 1 / torch.sqrt(nu2 + self.eps)
        y = self.gamma * x + self.beta
        z = torch.max(y, self.tau)
        return z

# Model from DeepOBS benchmark suite:
# https://github.com/fsschneider/DeepOBS/blob/develop/deepobs/pytorch/testproblems/testproblems_modules.py
class CIFAR10Net(nn.Sequential):
    """
    Deepobs network with optional last sigmoid activation (instead of relu)
    In Deepobs called `net_cifar10_3c3d`
    """
    def __init__(self, in_channels=3, n_out=10, use_tanh=False):
        super(CIFAR10Net, self).__init__()
        self.output_size = n_out
        activ = nn.Tanh if use_tanh else nn.ReLU

        self.add_module('conv1', tfconv2d(
            in_channels=in_channels, out_channels=64, kernel_size=5))
        self.add_module('relu1', nn.ReLU())
        self.add_module('maxpool1', tfmaxpool2d(
            kernel_size=3, stride=2, tf_padding_type='same'))

        self.add_module('conv2', tfconv2d(
            in_channels=64, out_channels=96, kernel_size=3))
        self.add_module('relu2', nn.ReLU())
        self.add_module('maxpool2', tfmaxpool2d(
            kernel_size=3, stride=2, tf_padding_type='same'))

        self.add_module('conv3', tfconv2d(
            in_channels=96, out_channels=128, kernel_size=3, tf_padding_type='same'))
        self.add_module('relu3', nn.ReLU())
        self.add_module('maxpool3', tfmaxpool2d(
            kernel_size=3, stride=2, tf_padding_type='same'))

        self.add_module('flatten', nn.Flatten())

        self.add_module('dense1', nn.Linear(
            in_features=3 * 3 * 128, out_features=512))
        self.add_module('relu4', activ())
        self.add_module('dense2', nn.Linear(in_features=512, out_features=256))
        self.add_module('relu5', activ())
        self.add_module('dense3', nn.Linear(in_features=256, out_features=n_out))

        # init the layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.constant_(module.bias, 0.0)
                nn.init.xavier_normal_(module.weight)

            if isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0.0)
                nn.init.xavier_uniform_(module.weight)

# Function from DeepOBS benchmark suite:
# https://github.com/fsschneider/DeepOBS/blob/develop/deepobs/pytorch/testproblems/testproblems_utils.py
def tfconv2d(in_channels,
             out_channels,
             kernel_size,
             stride=1,
             dilation=1,
             groups=1,
             bias=True,
             tf_padding_type=None
             ):
    modules = []
    if tf_padding_type == 'same':
        padding = nn.ZeroPad2d(0)
        hook = hook_factory_tf_padding_same(kernel_size, stride)
        padding.register_forward_pre_hook(hook)
        modules.append(padding)

    modules.append(nn.Conv2d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=0,
                             dilation=dilation,
                             groups=groups,
                             bias=bias))
    return nn.Sequential(*modules)

# Function from DeepOBS benchmark suite:
# https://github.com/fsschneider/DeepOBS/blob/develop/deepobs/pytorch/testproblems/testproblems_utils.py
def tfmaxpool2d(kernel_size,
                 stride=None,
                 dilation=1,
                 return_indices=False,
                 ceil_mode=False,
                 tf_padding_type = None):
    """
    Implements tf's padding 'same' for maxpooling
    """
    modules = []
    if tf_padding_type == 'same':
        padding = nn.ZeroPad2d(0)
        hook = hook_factory_tf_padding_same(kernel_size, stride)
        padding.register_forward_pre_hook(hook)
        modules.append(padding)

    modules.append(nn.MaxPool2d(kernel_size=kernel_size,
             stride=stride,
             padding=0,
             dilation=dilation,
             return_indices=return_indices,
             ceil_mode=ceil_mode,
             ))

    return nn.Sequential(*modules)

# Function from DeepOBS benchmark suite:
# https://github.com/fsschneider/DeepOBS/blob/develop/deepobs/pytorch/testproblems/testproblems_utils.py
def _determine_padding_from_tf_same(input_dimensions, kernel_dimensions, stride_dimensions):
    """
    Implements tf's padding 'same' for kernel processes like convolution or pooling.
    Args:
        input_dimensions (int or tuple): dimension of the input image
        kernel_dimensions (int or tuple): dimensions of the convolution kernel
        stride_dimensions (int or tuple): the stride of the convolution

     Returns: A padding 4-tuple for padding layer creation that mimics tf's padding 'same'.
     """

    # get dimensions
    in_height, in_width = input_dimensions

    if isinstance(kernel_dimensions, int):
        kernel_height = kernel_dimensions
        kernel_width = kernel_dimensions
    else:
        kernel_height, kernel_width = kernel_dimensions

    if isinstance(stride_dimensions, int):
        stride_height = stride_dimensions
        stride_width = stride_dimensions
    else:
        stride_height, stride_width = stride_dimensions

    # determine the output size that is to achive by the padding
    out_height = ceil(in_height / stride_height)
    out_width = ceil(in_width / stride_width)

    # determine the pad size along each dimension
    pad_along_height = max((out_height - 1) * stride_height + kernel_height - in_height, 0)
    pad_along_width = max((out_width - 1) * stride_width + kernel_width - in_width, 0)

    # determine padding 4-tuple (can be asymmetric)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return pad_left, pad_right, pad_top, pad_bottom

# Function from DeepOBS benchmark suite:
# https://github.com/fsschneider/DeepOBS/blob/develop/deepobs/pytorch/testproblems/testproblems_utils.py
def hook_factory_tf_padding_same(kernel_size, stride):
    """
    Implements tf's padding 'same' for maxpooling
    Generates the torch pre forward hook that needs to be registered on
    the padding layer to mimic tf's padding 'same'
    """
    def hook(module, input):
        """The hook overwrites the padding attribute of the padding layer."""
        image_dimensions = input[0].size()[-2:]
        module.padding = _determine_padding_from_tf_same(image_dimensions, kernel_size, stride)
    return hook

