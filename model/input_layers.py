import torch
from torch import nn
from torchvision.transforms.functional import center_crop

from model.utils import make_4d_1d

class BasicPatches(nn.Module):
    def __init__(self, patch_size):
        super(BasicPatches, self).__init__()
        self.patch_size = patch_size

    def split_into_patches(self, input):
        return input.unfold(-2, self.patch_size, self.patch_size).unfold(-2, self.patch_size, self.patch_size).contiguous()

    def forward(self, input):
        h_new = self.patch_size * (input.shape[-2] // self.patch_size) + self.patch_size * ((input.shape[-2] % self.patch_size) > 0)
        w_new = self.patch_size * (input.shape[-1] // self.patch_size) + self.patch_size * ((input.shape[-1] % self.patch_size) > 0)

        input = center_crop(input, (h_new, w_new))
        input = input.squeeze(0)
        input = self.split_into_patches(input)
        input = make_4d_1d(input)
        return input

class BasicConv(nn.Module):
    def __init__(self, patch_size, n_channels: int = 3, bias: bool = False):
        super(BasicConv, self).__init__()
        self.patch_size = patch_size
        self.input_conv = nn.Conv2d(n_channels, patch_size ** 2, (patch_size, patch_size), stride=patch_size, bias=bias)

    def forward(self, input):
        input = self.input_conv(input)
        input = input.transpose(-3, -1).contiguous()
        return input
