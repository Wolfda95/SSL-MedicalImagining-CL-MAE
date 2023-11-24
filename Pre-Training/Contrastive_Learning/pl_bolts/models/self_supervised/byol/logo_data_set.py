from logging import getLogger

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image, ImageFilter

logger = getLogger()


class LoGoDataset(datasets.ImageFolder):
    def __init__(
        self,
        data_path,
        tau_g,
        tau_l,
    ):
        super(LoGoDataset, self).__init__(data_path)
        self.tau_g = tau_g  # set of global transforms
        self.tau_l = tau_l  # set of local transforms

    def __len__(self):
        return len(self.samples)

    def __str__(self):
        return f"LoGoDataset with {self.__len__()} images"

    def __getitem__(self, idx):
        path, _ = self.samples[idx]
        image = self.loader(path)

        global_crops = list(map(lambda transform: transform(image), self.tau_g))
        local_crops = list(map(lambda transform: transform(image), self.tau_l))

        # breakpoint()

        return (global_crops[0], global_crops[1]), 0
