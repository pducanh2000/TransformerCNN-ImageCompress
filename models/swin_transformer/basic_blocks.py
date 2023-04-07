import torch
import torch.nn as nn
from utils.helper_functions import drop_path


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self) -> str:
        return f'drop_prob={round(self.drop_prob, 3): 0.3f}'


class ResidualBlock(nn.Module):
    """
    Simple residual block with 2 3x3 convolutions.
    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output_channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super(ResidualBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.conv1 = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=3, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.out_ch, self.out_ch, kernel_size=3, stride=1, padding=1)

        self.skip_conv = None
        if self.in_ch != self.out_ch:
            self.skip_conv = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        if self.skip_conv is not None:
            identity = self.skip_conv(x)

        return out + identity


class AttentionBlock(nn.Module):
    """
    Self attention block
    Args:
        N (int): number of channels
    """
    def __init__(self, N):
        super(AttentionBlock, self).__init__()

        class ResidualUnit(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(N, N//2, kernel_size=1, stride=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(N//2, N//2, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(N//2, N, kernel_size=1, stride=1),
                )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x: torch.Tensor):
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out
        self.conv_a = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit()
        )

        self.conv_b = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            nn.Conv2d(N, N, kernel_size=1, stride=1)
        )

    def forward(self, x):
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity

        return out
