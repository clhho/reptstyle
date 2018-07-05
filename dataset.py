from glob import glob
from os import path as osp

import torch
from PIL import Image
from torchvision import transforms


class ImageFolder(torch.utils.data.Dataset):
    def __init__(self, folder, transform=None):
        super().__init__()

        self.images = sorted(glob(osp.join(folder, '*')))

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        return self.transform(Image.open(self.images[index]).convert('RGB'))

    def __len__(self):
        return len(self.images)


def test_dataset():
    txform = transforms.Compose([
        transforms.Scale(256),
        transforms.RandomCrop(256),
        transforms.ToTensor(),
    ])
    ds = ImageFolder('data/coco/', txform)
    print(ds[0].shape)
