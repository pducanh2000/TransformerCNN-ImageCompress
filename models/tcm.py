import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch import Tensor

from einops import rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_, DropPath


SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

