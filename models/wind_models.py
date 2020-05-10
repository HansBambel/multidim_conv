import torch
import torch.nn.functional as F
from torch import nn
from models.attention_augmented_conv import AugmentedConv
from models.layers import DoubleDense, DepthwiseSeparableConv, DoubleDSConv
from einops import rearrange


class CNN2DWind_DK(nn.Module):
    def __init__(self, in_channels, output_channels, feature_maps, hidden_neurons=128):
        super(CNN2DWind_DK, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=feature_maps, kernel_size=2)
        # Correctly calculate input features (depends on conv2d)
        # ((W−F+2P)/S)+1 --> ((4-2+0)/1)+1 = 3
        self.dd = DoubleDense(feature_maps * 3 * 3, hidden_neurons=hidden_neurons, output_channels=output_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # Maybe flatten
        x = self.dd(x.view(x.size(0), -1))
        return x


class CNN2DAttWind_DK(nn.Module):
    def __init__(self, in_channels, output_channels, feature_maps, hidden_neurons=128):
        super(CNN2DAttWind_DK, self).__init__()
        self.conv1 = AugmentedConv(in_channels=in_channels, out_channels=feature_maps, kernel_size=2,
                                   dk=feature_maps//2, dv=feature_maps//2, Nh=feature_maps//2, shape=0, relative=False, stride=1)
        # Correctly calculate input features (depends on conv2d)
        # ((W−F+2P)/S)+1 --> ((4-2+0)/1)+1 = 3
        self.dd = DoubleDense(feature_maps * 3 * 3, hidden_neurons=hidden_neurons, output_channels=output_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # Maybe flatten
        x = self.dd(x.view(x.size(0), -1))
        return x


class CNNDS2DDeconvWind_DK(nn.Module):
    def __init__(self, in_channels, output_channels, feature_maps, hidden_neurons=128):
        super(CNNDS2DDeconvWind_DK, self).__init__()
        # scale up input and then use a bigger kernel size to increase receptive field
        self.upscale = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        # 5x8x8
        self.doubleDSConv = DoubleDSConv(in_channels, feature_maps)
        self.dsConv1 = DepthwiseSeparableConv(feature_maps, output_channels=feature_maps // 2, kernel_size=3,
                                              padding=0)
        # ((W−F+2P)/S)+1 --> (8x8)-3+1 = (6x6)
        self.dsConv2 = DepthwiseSeparableConv(feature_maps // 2, output_channels=feature_maps // 4, kernel_size=3,
                                              padding=0)
        # (6x6)-3+1 = (4x4)
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
        self.conv1 = AugmentedConv(in_channels=in_channels, out_channels=feature_maps, kernel_size=2, dk=feature_maps//2, dv=feature_maps//2,
                                   Nh=feature_maps//2, shape=0, relative=False, stride=1)
        # Correctly calculate input features (depends on conv2d)
        # ((W−F+2P)/S)+1 --> (((6x6)-2+0)/1)+1 = (5x5)
        self.dd = DoubleDense(feature_maps * 5 * 5, hidden_neurons=hidden_neurons, output_channels=output_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # Maybe flatten
        x = self.dd(x.view(x.size(0), -1))
        return x


class CNNDS2DDeconvWind_NL(nn.Module):
    def __init__(self, in_channels, output_channels, feature_maps, hidden_neurons=128):
        super(CNNDS2DDeconvWind_NL, self).__init__()
        # scale up input and then use a bigger kernel size to increase receptive field
        self.upscale = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        # 5x12x12
        self.doubleDSConv = DoubleDSConv(in_channels, feature_maps)
        # Depthwise separable convolution
        self.dsConv1 = DepthwiseSeparableConv(feature_maps, output_channels=feature_maps // 2, kernel_size=3, padding=0)
        # ((W−F+2P)/S)+1 --> (14x12)-3+1 = (10x10)
        self.dsConv2 = DepthwiseSeparableConv(feature_maps // 2, output_channels=feature_maps // 4, kernel_size=3,
                                              padding=0)
        # (10x10)-3+1 = (8x8)
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


class CNN3DWind_DK(nn.Module):
    def __init__(self, in_channels, output_channels, feature_maps, hidden_neurons=128):
        super(CNN3DWind_DK, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=feature_maps, kernel_size=(2, 2, 2))
        # Correctly calculate input features (depends on conv3d)
        # ((W−F+2P)/S)+1 --> 5-2+1 ((4-2+0)/1)+1 = 4*3*3
        self.dd = DoubleDense(feature_maps * 4 * 3 * 3, hidden_neurons=hidden_neurons, output_channels=output_channels)

    def forward(self, x):
        x = F.relu(self.conv1(torch.unsqueeze(x, 1)))
        x = self.dd(x.view(x.size(0), -1))
        return x


class CNN3DWind_NL(nn.Module):
    def __init__(self, in_channels, output_channels, feature_maps, hidden_neurons=128):
        super(CNN3DWind_NL, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=feature_maps, kernel_size=(2, 2, 2))
        # Correctly calculate input features (depends on conv3d)
        # ((W−F+2P)/S)+1 --> (7x6x6)-2+1 = 6x5x5
        self.dd = DoubleDense(feature_maps * 6 * 5 * 5, hidden_neurons=hidden_neurons, output_channels=output_channels)

    def forward(self, x):
        x = F.relu(self.conv1(torch.unsqueeze(x, 1)))
        x = self.dd(x.view(x.size(0), -1))
        return x


class MultidimConv(nn.Module):
    def __init__(self, channels, height, width, kernel_size=3, kernels_per_layer=16, padding=1):
        super(MultidimConv, self).__init__()
        self.normal = DepthwiseSeparableConv(channels, channels, kernels_per_layer=kernels_per_layer,
                                             kernel_size=kernel_size, padding=padding)
        self.horizontal = DepthwiseSeparableConv(height, height, kernels_per_layer=kernels_per_layer,
                                                 kernel_size=kernel_size, padding=padding)
        self.vertical = DepthwiseSeparableConv(width, width, kernels_per_layer=kernels_per_layer,
                                               kernel_size=kernel_size, padding=padding)
        self.bn_normal = nn.BatchNorm2d(channels)
        self.bn_horizontal = nn.BatchNorm2d(height)
        self.bn_vertical = nn.BatchNorm2d(width)

    def forward(self, x):
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
    def __init__(self, channels, height, width, output_channels, kernel_size=3, kernels_per_layer=16, padding=1,
                 hidden_neurons=128):
        super(MultidimConvNetwork, self).__init__()
        self.multidim = MultidimConv(channels, height, width, kernel_size=kernel_size,
                                     kernels_per_layer=kernels_per_layer, padding=padding)

        self.merge = nn.Linear(3 * channels * width * height, hidden_neurons)
        self.output = nn.Linear(hidden_neurons, output_channels)

    def forward(self, x):
        output = self.multidim(x)
        output = F.relu(self.merge(output))
        output = self.output(output)
        return output