import random
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import ImageFilter
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from moco_data_set import MoCoDataset


class MoCoDataModule(LightningDataModule):
    """Example of LightningDataModule. A DataModule implements 5 key methods:

        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))
    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = True,
        shuffle: bool = True,
        global_crop_size: int = 224,
        global_crop_scale: Tuple[float, float] = (0.2, 1.0),
        local_crop_size: int = 64,
        local_crop_scale: Tuple[float, float] = (0.5, 1.0),
        drop_last: bool = True,
        mean: list = None,
        std: list = None,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.global_crop_size = global_crop_size
        self.global_crop_scale = global_crop_scale
        self.local_crop_size = local_crop_size
        self.local_crop_scale = local_crop_scale
        self.drop_last = drop_last
        self.mean = mean
        self.std = std

        normalize = transforms.Normalize(mean=self.mean, std=self.std)

        # MoCov2 image transforms
        self.tau_g = [
            transforms.Compose(
                [
                    transforms.Grayscale(3),
                    transforms.RandomApply([transforms.RandomRotation(180)], p=0.5),
                    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                    transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                    # transforms.Grayscale(1),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    GaussNoise(p=0.5),
                    normalize,
                ]
            )
        ] * 2  # make 2 global crops from one image

        self.tau_l = [
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(self.local_crop_size, scale=self.local_crop_scale),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
                    transforms.RandomGrayscale(p=0.2),
                    # transforms.RandomApply([moco.loader.GaussianBlur([0.1, 2.0])], p=0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        ] * 2

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        """Load data.

        Set variables: self.data_train, self.data_val, self.data_test.
        """
        if stage == "fit":
            self.train_set = MoCoDataset(self.data_dir, self.tau_g)
        else:
            print("Not implemented")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
        )


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class GaussNoise:
    """Gaussian Noise to be applied to images that have been scaled to fit in the range 0-1"""

    def __init__(self, var_limit=(1e-5, 1e-4), p=0.5):
        self.var_limit = np.log(var_limit)
        self.p = p

    def __call__(self, image):
        if np.random.random() < self.p:
            sigma = np.exp(np.random.uniform(*self.var_limit)) ** 0.5
            noise = np.random.normal(0, sigma, size=image.shape).astype(np.float32)
            image = image + torch.from_numpy(noise)
            image = torch.clamp(image, 0, 1)

        return image


if __name__ == "__main__":

    # testing the data module and data set
    dm = MoCoDataModule("/mnt/hdd/datasets/Imagenet100/train")
    dm.setup(stage="fit")
    dl = dm.train_dataloader()
    imgs = next(iter(dl))
    breakpoint()
