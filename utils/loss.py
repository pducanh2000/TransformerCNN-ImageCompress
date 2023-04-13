import math
import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim


def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.)


class RateDistortionLoss(nn.Module):
    """
    Custom rate distortion loss with a Lagrangian parameter.
    """

    def __init__(self, lmbda=1e-2, type="mse"):
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.type = type

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        if self.type == "mse":
            out["mse_loss"] = self.mse(output["x_hat"], target)
            out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        else:
            out["ms_ssim_loss"] = compute_msssim(output["x_hat"], target)
            out["loss"] = self.lmbda * (1 - out["ms_ssim_loss"]) + out["bpp_loss"]

        return out


class AverageMeter:
    """
    Compute running average
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

