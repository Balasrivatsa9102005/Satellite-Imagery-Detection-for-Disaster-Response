import os
import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import torch
from torch.utils.data import Dataset

import torchvision.transforms as transforms


class LEVIRDataset(Dataset):
    """
    LEVIR-CD Dataset Loader

    Returns:
        before : Tensor (3, 256, 256)
        after  : Tensor (3, 256, 256)
        mask   : Tensor (256, 256)
    """

    def __init__(self, root_dir, transform=None):

        self.root_dir = Path(root_dir)
        self.transform = transform

        self.a_images = sorted((self.root_dir / "A").glob("*"))
        self.b_images = sorted((self.root_dir / "B").glob("*"))
        self.labels = sorted((self.root_dir / "label").glob("*"))

    def __len__(self):
        return len(self.a_images)

    def __getitem__(self, index):

        before = np.array(
            Image.open(self.a_images[index]).convert("RGB")
        )

        after = np.array(
            Image.open(self.b_images[index]).convert("RGB")
        )

        mask = np.array(
            Image.open(self.labels[index])
        )

        # Binary mask
        mask = (mask > 0).astype(np.uint8)

        # Resize mask using nearest-neighbor interpolation
        mask = Image.fromarray(mask)
        mask = mask.resize((256, 256), Image.NEAREST)
        mask = np.array(mask).astype(np.float32)

        # Apply transforms to images
        if self.transform:
            before = self.transform(before)
            after = self.transform(after)

        # Convert mask to tensor
        mask = torch.tensor(mask, dtype=torch.float32)

        return before, after, mask


# -------------------------------------------------------
# Image Transforms
# -------------------------------------------------------

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.45026043, 0.4466681, 0.38134657],
        std=[0.19258917, 0.18164736, 0.16908678]
    )
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.45026043, 0.4466681, 0.38134657],
        std=[0.19258917, 0.18164736, 0.16908678]
    )
])