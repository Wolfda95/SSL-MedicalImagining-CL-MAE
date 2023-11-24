# von Facbook Code übernommen


# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
from logging import getLogger

import numpy as np
from skimage.util import random_noise
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image, ImageFilter

logger = getLogger()


class MultiCropDataset(datasets.ImageFolder):
    def __init__(
            self,
            data_path,
            size_crops,
            nmb_crops,
            min_scale_crops,
            max_scale_crops,
            size_dataset=-1,
            return_index=False,
    ):
        super(MultiCropDataset, self).__init__(data_path)
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        self.return_index = return_index

        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]

        # Covid
        # mean = [0.3091, 0.3091, 0.3091]
        # std = [0.2023, 0.2023, 0.2023]

        # All Data: Similar
        mean = [0.1806, 0.1806, 0.1806]
        std = [0.1907, 0.1907, 0.1907]

        # All Data: not Similar
        # mean = [0.1695, 0.1695, 0.1695]
        # std = [0.1826, 0.1826, 0.1826]

        # ImageNet
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]

        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([transforms.Compose([
                # transforms.ToTensor(),
                randomresizedcrop,

                transforms.Compose(color_transform),

                # transforms.ColorJitter( # NEU Randomly change the [brightness, contrast, saturation, hue] of an image
                #    brightness = 0.4,
                #    contrast = 0.4,
                #    saturation = 0.4,
                #    hue = 0.1
                # ),

                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomVerticalFlip(p=0.5), #NEU

                # transforms.Compose(color_transform),

                # PoissonNoise_Mogl1(),

                transforms.ToTensor(),

                # PoissonNoise_Mogl2(factor=300),

                transforms.Normalize(mean=mean, std=std)

            ])
                         ] * nmb_crops[i])
        self.trans = trans

    def __getitem__(self, index):
        path, _ = self.samples[index]
        image = self.loader(path)
        multi_crops = list(map(lambda trans: trans(image), self.trans))
        if self.return_index:
            return index, multi_crops
        return multi_crops


class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class PoissonNoise_Mogl2(object):
    def __init__(self, factor=1):
        self.factor = factor

    def __call__(self, tensor):
        return tensor + torch.poisson(tensor * self.factor)


class PoissonNoise_Mogl1(object):
    def __init__(self):
        pass

    def __call__(self, img):
        noisy_img = random_noise(np.array(img), mode="poisson")
        return Image.fromarray(np.uint8(noisy_img * 255))


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.4 * s, 0.4 * s, 0.2 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort