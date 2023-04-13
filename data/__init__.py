import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class CustomDataParallel(nn.DataParallel):
    """
    Custom parallel to access the module method
    """
    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return getattr(self.module, item)
