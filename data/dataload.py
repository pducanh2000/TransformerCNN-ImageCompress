from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self, args, mode="train", transform=True):
        self.args = args
        self.data = glob(self.args.path + "*.jpg")
        self.mode = mode
        self.transform = transform

    def make_transformations(self, image):
        if self.mode == "train":
            changes = transforms.Compose(
                [
                    transforms.RandomCrop(self.args.patch_size),
                    transforms.ToTensor()
                ]
            )
        elif self.mode == "test":
            changes = transforms.Compose(
                [
                    transforms.CenterCrop(self.args.patch_size),
                    transforms.ToTensor()
                ]
            )
        else:
            changes = transforms.Compose(
                [
                    transforms.Resize(self.args.patch_size),
                    transforms.ToTensor()
                ]
            )
        return changes(image)

    def __getitem__(self, item):
        image = self.data[item]
        if self.transform:
            image = self.make_transformations(image)
        else:
            image = torch.Tensor(image)
        return image

    def __len__(self):
        return len(self.data)
