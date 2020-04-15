import torch
import torch.nn.functional as F
from torch import nn
from models.attention_augmented_conv import AugmentedConv
from einops import rearrange


class CNN2DWind(nn.Module):
    def __init__(self, in_channels, output_channels, feature_maps, hidden_neurons=128):
        super(CNN2DWind, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=feature_maps, kernel_size=2)
        # Correctly calculate input features (depends on conv2d)
        # ((W−F+2P)/S)+1 --> ((4-2+0)/1)+1 = 3
        self.dd = DoubleDense(feature_maps * 3 * 3, hidden_neurons=hidden_neurons, output_channels=output_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # Maybe flatten
        x = self.dd(x.view(x.size(0), -1))
        return x


class CNN2DAttWind(nn.Module):
    def __init__(self, in_channels, output_channels, feature_maps, hidden_neurons=128):
        super(CNN2DAttWind, self).__init__()
        self.conv1 = AugmentedConv(in_channels=in_channels, out_channels=feature_maps, kernel_size=2, dk=16, dv=16, Nh=16, shape=0, relative=False, stride=1)
        # Correctly calculate input features (depends on conv2d)
        # ((W−F+2P)/S)+1 --> ((4-2+0)/1)+1 = 3
        self.dd = DoubleDense(feature_maps * 3 * 3, hidden_neurons=hidden_neurons, output_channels=output_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # Maybe flatten
        x = self.dd(x.view(x.size(0), -1))
        return x


class CNN3DWind(nn.Module):
    def __init__(self, in_channels, output_channels, feature_maps, hidden_neurons=128):
        super(CNN3DWind, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=feature_maps, kernel_size=(2, 2, 2))
        # Correctly calculate input features (depends on conv3d)
        # ((W−F+2P)/S)+1 --> 5-2+1 ((4-2+0)/1)+1 = 4*3*3
        self.dd = DoubleDense(feature_maps * 4 * 3 * 3, hidden_neurons=hidden_neurons, output_channels=output_channels)

    def forward(self, x):
        x = F.relu(self.conv1(torch.unsqueeze(x, 1)))
        x = self.dd(x.view(x.size(0), -1))
        return x


class CNN2DDepthwiseSeparable(nn.Module):
    """
        When groups == in_channels and out_channels == K * in_channels, where K is a positive integer,
        this operation is also termed in literature as depthwise convolution.
        In other words, for an input of size (N, C_{in}, H_{in}, W_{in})(N,Cin,Hin,Win),
        a depthwise convolution with a depthwise multiplier K, can be constructed by arguments
        (in_channels=Cin, out_channels=Cin x K, ..., groups=Cin)
    """

    def __init__(self, in_channels, feature_maps, output_channels, hidden_neurons=128):
        super(CNN2DDepthwiseSeparable, self).__init__()
        self.dsConv = DepthwiseSeparableConv(in_channels, feature_maps, 2, 0)
        # ((W−F+2P)/S)+1 --> feature_mapsx3x3
        self.dd = DoubleDense(feature_maps * 3 * 3, hidden_neurons=hidden_neurons, output_channels=output_channels)

    def forward(self, x):
        out = F.relu(self.dsConv(x))
        out = self.dd(out.view(out.size(0), -1))
        return out


class DoubleDense(nn.Module):
    def __init__(self, in_channels, hidden_neurons, output_channels):
        super(DoubleDense, self).__init__()
        self.dense1 = nn.Linear(in_channels, out_features=hidden_neurons)
        self.dense2 = nn.Linear(in_features=hidden_neurons, out_features=hidden_neurons // 2)
        self.dense3 = nn.Linear(in_features=hidden_neurons // 2, out_features=output_channels)

    def forward(self, x):
        out = F.relu(self.dense1(x.view(x.size(0), -1)))
        out = F.relu(self.dense2(out))
        out = self.dense3(out)
        return out


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size, padding):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DoubleDSConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_ds_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_ds_conv(x)


class CNNDW2DDeconvWind(nn.Module):
    def __init__(self, in_channels, output_channels, feature_maps, hidden_neurons=128):
        super(CNNDW2DDeconvWind, self).__init__()
        # scale up input and then use a bigger kernel size to increase receptive field
        self.upscale = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        # 5x8x8
        self.doubleDSConv = DoubleDSConv(in_channels, feature_maps)
        # Depthwise separable convolution
        self.dsConv1 = DepthwiseSeparableConv(feature_maps, output_channels=feature_maps // 2, kernel_size=3, padding=0)
        # ((W−F+2P)/S)+1 --> 8-3+1 = 6x6
        self.dsConv2 = DepthwiseSeparableConv(feature_maps // 2, output_channels=feature_maps // 4, kernel_size=3,
                                              padding=0)
        # 6-3+1 = 4
        self.bn = nn.BatchNorm2d(feature_maps // 4)

        self.doubleDense = DoubleDense(feature_maps // 4 * 4 * 4, hidden_neurons=hidden_neurons,
                                       output_channels=output_channels)

    def forward(self, x):
        out = self.upscale(x)
        out = self.doubleDSConv(out)
        out = F.relu(self.dsConv1(out))
        out = F.relu(self.dsConv2(out))
        out = self.bn(out)
        out = self.doubleDense(out)
        return out


class DepthwiseSeparableMultikernelsConv(nn.Module):
    def __init__(self, in_channels, output_channels, kernels_per_layer, kernel_size, padding):
        super(DepthwiseSeparableMultikernelsConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels * kernels_per_layer, kernel_size=kernel_size,
                                   padding=padding,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MultidimConv(nn.Module):
    def __init__(self, channels, height, width, kernels_per_layer=16, kernel_size=3, padding=1):
        super(MultidimConv, self).__init__()
        self.normal = DepthwiseSeparableMultikernelsConv(channels, channels, kernels_per_layer=kernels_per_layer,
                                                         kernel_size=kernel_size, padding=padding)
        self.horizontal = DepthwiseSeparableMultikernelsConv(height, height, kernels_per_layer=kernels_per_layer,
                                                             kernel_size=kernel_size, padding=padding)
        self.vertical = DepthwiseSeparableMultikernelsConv(width, width, kernels_per_layer=kernels_per_layer,
                                                           kernel_size=kernel_size, padding=padding)
        self.bn_normal = nn.BatchNorm2d(channels)
        self.bn_horizontal = nn.BatchNorm2d(height)
        self.bn_vertical = nn.BatchNorm2d(width)

    def forward(self, x):
        ### First Block
        x_normal = self.normal(x)
        x_horizontal = self.horizontal(rearrange(x, "b c h w -> b h c w"))  # x.permute(0,2,1,3)
        x_vertical = self.vertical(rearrange(x, "b c h w -> b w c h"))  # x.permute(0,3,1,2)
        x_normal = F.relu(self.bn_normal(x_normal))
        x_horizontal = F.relu(self.bn_horizontal(x_horizontal))
        x_vertical = F.relu(self.bn_vertical(x_vertical))
        output = torch.cat([rearrange(x_normal, "b c h w -> b (c h w)"),
                            rearrange(x_horizontal, "b c h w -> b (c h w)"),
                            rearrange(x_vertical, "b c h w -> b (c h w)")
                            ], dim=1)
        return output


class MultidimConvNetwork(nn.Module):
    def __init__(self, channels, height, width, output_channels, kernels_per_layer=16, kernel_size=3, padding=1,
                 hidden_neurons=128):
        super(MultidimConvNetwork, self).__init__()
        self.multidimConv = MultidimConv(channels, height, width, kernels_per_layer=kernels_per_layer, kernel_size=kernel_size, padding=padding)

        self.merge = nn.Linear(3 * channels * width * height, hidden_neurons)
        self.output = nn.Linear(hidden_neurons, output_channels)

    def forward(self, x):
        output = self.multidimConv(x)
        output = F.relu(self.merge(output))
        output = self.output(output)
        return output


class CNN3DWind_NL(nn.Module):
    def __init__(self, in_channels, output_channels, feature_maps, hidden_neurons=128):
        super(CNN3DWind_NL, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=feature_maps, kernel_size=(2, 2, 2))
        # Correctly calculate input features (depends on conv3d)
        # ((W−F+2P)/S)+1 --> 5-2+1 ((4-2+0)/1)+1 = 565
        self.dd = DoubleDense(feature_maps * 5 * 6 * 5, hidden_neurons=hidden_neurons, output_channels=output_channels)

    def forward(self, x):
        x = F.relu(self.conv1(torch.unsqueeze(x, 1)))
        x = self.dd(x.view(x.size(0), -1))
        return x


### The following models are for data in the CitiesxTimestepxFeature format
class CNN2DWind_NL(nn.Module):
    def __init__(self, in_channels, output_channels, feature_maps, hidden_neurons=128):
        super(CNN2DWind_NL, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=feature_maps, kernel_size=2)
        # Correctly calculate input features (depends on conv2d)
        # ((W−F+2P)/S)+1 --> (((6x6)-2+0)/1)+1 = (5x5)
        self.dd = DoubleDense(feature_maps * 5 * 5, hidden_neurons=hidden_neurons, output_channels=output_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # Maybe flatten
        x = self.dd(x.view(x.size(0), -1))
        return x


class CNN2DAttWind_NL(nn.Module):
    def __init__(self, in_channels, output_channels, feature_maps, hidden_neurons=128):
        super(CNN2DAttWind_NL, self).__init__()
        self.conv1 = AugmentedConv(in_channels=in_channels, out_channels=feature_maps, kernel_size=2, dk=16, dv=16, Nh=16, shape=0, relative=False, stride=1)
        # Correctly calculate input features (depends on conv2d)
        # ((W−F+2P)/S)+1 --> (((6x6)-2+0)/1)+1 = (5x5)
        self.dd = DoubleDense(feature_maps * 5 * 5, hidden_neurons=hidden_neurons, output_channels=output_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # Maybe flatten
        x = self.dd(x.view(x.size(0), -1))
        return x


class CNNDW2DDeconvWind_NL(nn.Module):
    def __init__(self, in_channels, output_channels, feature_maps, hidden_neurons=128):
        super(CNNDW2DDeconvWind_NL, self).__init__()
        # scale up input and then use a bigger kernel size to increase receptive field
        self.upscale = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        # 5x12x12
        self.doubleDSConv = DoubleDSConv(in_channels, feature_maps)
        # Depthwise separable convolution
        self.dsConv1 = DepthwiseSeparableConv(feature_maps, output_channels=feature_maps // 2, kernel_size=3,
                                              padding=0)
        # ((W−F+2P)/S)+1 --> (12x12)-3+1 = (10x10)
        self.dsConv2 = DepthwiseSeparableConv(feature_maps // 2, output_channels=feature_maps // 4, kernel_size=3,
                                              padding=0)
        # (12x10)-3+1 = (8x8)
        self.bn = nn.BatchNorm2d(feature_maps // 4)

        self.doubleDense = DoubleDense(feature_maps // 4 * 8 * 8, hidden_neurons=hidden_neurons,
                                       output_channels=output_channels)

    def forward(self, x):
        out = self.upscale(x)
        out = self.doubleDSConv(out)
        out = F.relu(self.dsConv1(out))
        out = F.relu(self.dsConv2(out))
        out = self.bn(out)
        out = self.doubleDense(out)
        return out