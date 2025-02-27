import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv3d(inChannels, outChannels, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm3d(outChannels)
        self.conv2 = nn.Conv3d(outChannels, outChannels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm3d(outChannels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class DeConvLayer(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(DeConvLayer, self).__init__()
        self.deconv = nn.ConvTranspose3d(inChannels, outChannels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm3d(outChannels)

    def forward(self, x):
        out = F.relu(self.bn(self.deconv(x)))
        return out


class maskConvAutoEncoder(nn.Module):
    def __init__(self, hidden_size):
        super(maskConvAutoEncoder, self).__init__()
        # encoder
        self.Conv1 = ConvBlock(1, hidden_size // 8)
        self.Conv2 = ConvBlock(hidden_size // 8, hidden_size // 4)
        self.Conv3 = ConvBlock(hidden_size // 4, hidden_size // 2)
        self.Conv4 = ConvBlock(hidden_size // 2, hidden_size)
        self.AvgPool = nn.AvgPool3d(kernel_size=2, stride=2)

        # decoder
        self.deConv1 = DeConvLayer(hidden_size, hidden_size // 2)
        self.Conv5 = ConvBlock(hidden_size // 2, hidden_size // 2)
        self.deConv2 = DeConvLayer(hidden_size // 2, hidden_size // 4)
        self.Conv6 = ConvBlock(hidden_size // 4, hidden_size // 4)
        self.deConv3 = DeConvLayer(hidden_size // 4, hidden_size // 8)
        self.Conv7 = ConvBlock(hidden_size // 8, hidden_size // 8)
        self.Conv8 = nn.Conv3d(hidden_size // 8, 1, kernel_size=1)

    def forward(self, x):
        feature = self.AvgPool(self.Conv1(x))
        feature = self.AvgPool(self.Conv2(feature))
        feature = self.AvgPool(self.Conv3(feature))
        feature = self.deConv1(self.Conv4(feature))
        feature = self.deConv2(self.Conv5(feature))
        feature = self.deConv3(self.Conv6(feature))
        out = self.Conv8(self.Conv7(feature))
        return out


class Feature_Extraction(nn.Module):
    def __init__(self, hidden_size):
        super(Feature_Extraction, self).__init__()
        # encoder for feature extraction
        self.Conv1 = ConvBlock(1, hidden_size // 8)
        self.Conv2 = ConvBlock(hidden_size // 8, hidden_size // 4)
        self.Conv3 = ConvBlock(hidden_size // 4, hidden_size // 2)
        self.Conv4 = ConvBlock(hidden_size // 2, hidden_size)
        self.AvgPool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        feature = self.AvgPool(self.Conv1(x))
        feature = self.AvgPool(self.Conv2(feature))
        out_m = self.Conv3(feature)
        feature = self.AvgPool(out_m)
        out_s = self.Conv4(feature)
        return out_s, out_m
