import torch
import torch.nn as nn
import torch.nn.functional as f


class WMSA(nn.Module):
    """
    Self attention module in SwinTransformer
    """

    def __init__(self, input_dim):