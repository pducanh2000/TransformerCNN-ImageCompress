import os
import argparse
import torch
from compressai.zoo import models


def save_checkpoint(state_dict, is_best, epoch, save_path, file_name):
    torch.save(state_dict, os.path.join(save_path, "checkpoint_last.pth"))
    if epoch % 5 == 0:
        torch.save(state_dict, file_name)
    if is_best:
        torch.save(state_dict, os.path.join(save_path, "checkpoint_best.pth"))


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "-m", "--model", default="bmshj2018-factorized", choices=models.keys(), help="Model Architecture"
    )
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Training dataset")
    parser.add_argument("-e", "--epoch", type=int, default=50, help="Number of epochs")
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("-n", "--num-workers", type=int, default=4, help="Number of data loader threads")
    parser.add_argument("--lambda", type=float, default=3, help="Rate-Distortion weight")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--test-batch-size", type=int, default=8, help="Batch size in testing")
    parser.add_argument("--aux-learning-rate", type=float, default=1e-3, help="Auxiliary loss learning rate")
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256), help="Size of path to be cropped")
    parser.add_argument("--save", action="store_true", default=True, help="Save model")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--clip_max_norm", type=float, default=1.0, help="Gradient clipping max norm")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "ms-ssim"], help="Loss type")
    parser.add_argument("--save_path", type=str, help="Save path")
    parser.add_argument("--skip_epoch", type=int, default=0)
    parser.add_argument("--N", type=int, default=128, help="N = Transformer dimension = Conv dimension")
    parser.add_argument("--lr_epoch", nargs="+", type=int)
    parser.add_argument("--continue_train", action="store_true", default=True)

    args = parser.parse_args(argv)
    return args
