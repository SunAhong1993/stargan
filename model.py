from x2paddle import torch2paddle
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np


class ResidualBlock(nn.Layer):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(nn.Conv2D(dim_in, dim_out, kernel_size=3,
            stride=1, padding=1, bias_attr=False), nn.InstanceNorm2D(
            dim_out), nn.ReLU(), nn.Conv2D(dim_out,
            dim_out, kernel_size=3, stride=1, padding=1, bias_attr=False),
            nn.InstanceNorm2D(dim_out))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Layer):
    """Generator network."""

    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()
        layers = []
        layers.append(nn.Conv2D(3 + c_dim, conv_dim, kernel_size=7, stride=\
            1, padding=3, bias_attr=False))
        layers.append(nn.InstanceNorm2D(conv_dim))
        layers.append(nn.ReLU())
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2D(curr_dim, curr_dim * 2, kernel_size=4,
                stride=2, padding=1, bias_attr=False))
            layers.append(nn.InstanceNorm2D(curr_dim * 2))
            layers.append(nn.ReLU())
            curr_dim = curr_dim * 2
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        for i in range(2):
            layers.append(torch2paddle.Conv2DTranspose(curr_dim, curr_dim //
                2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2D(curr_dim // 2))
            layers.append(nn.ReLU())
            curr_dim = curr_dim // 2
        layers.append(nn.Conv2D(curr_dim, 3, kernel_size=7, stride=1,
            padding=3, bias_attr=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch2paddle.concat([x, c], dim=1)
        return self.main(x)


class Discriminator(nn.Layer):
    """Discriminator network with PatchGAN."""

    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2D(3, conv_dim, kernel_size=4, stride=2,
            padding=1))
        layers.append(nn.LeakyReLU(0.01))
        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2D(curr_dim, curr_dim * 2, kernel_size=4,
                stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2
        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2D(curr_dim, 1, kernel_size=3, stride=1,
            padding=1, bias_attr=False)
        self.conv2 = nn.Conv2D(curr_dim, c_dim, kernel_size=kernel_size,
            bias_attr=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
