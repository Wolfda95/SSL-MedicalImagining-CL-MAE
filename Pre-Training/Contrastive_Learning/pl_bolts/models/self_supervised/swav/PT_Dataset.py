import os
import random
import re

import numpy as np
import torch
from torch.utils.data import Dataset


class TorchDataset(Dataset):
    """
    Loading the Datasets
    """

    def __init__(self, directory, augmentations=False):
        self.directory = directory # 1)
        self.augmentations = augmentations

        self.images = os.listdir(directory) # 2) Liste von allen Files in directory (alle Images)


    # def augment_gaussian_noise(self, data_sample, noise_variance=(0.001, 0.05)):
    #     # https://github.com/MIC-DKFZ/batchgenerators
    #     if noise_variance[0] == noise_variance[1]:
    #         variance = noise_variance[0]
    #     else:
    #         variance = random.uniform(noise_variance[0], noise_variance[1])
    #     data_sample = data_sample + np.random.normal(0.0, variance, size=data_sample.shape)
    #     return data_sample

    def __len__(self):
        return len(os.listdir(self.directory)) # 3) Anzahl an Files in directory (Anzahl Bilder)

    def __getitem__(self, idx): # idx = Anzahl Aufrufe (Iteration 0,1,2,3,...)
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # load image + lable
        name = self.images[idx] # 5) image: Liste von allen Files in directory (alle Images) [idx: geht Liste durch, eins Laden pro Aufruf]
        file = torch.load(os.path.join(self.directory, name)) # 6) Läd ein File pro Aufruf


        # Image / Lable trennen
        image = file["vol"]
        lable = file["class"]


        # Falls nicht als float sondern als Torch Tensor abgespeichert:
        image = image.to(torch.float32)


        # do augmentations
        if self.augmentations:
            random_number = random.randint(1, 10)
            image = image.numpy()
            if random_number >= 7:
                # do for each layer
                image = self.augment_gaussian_noise(image)
            image = torch.from_numpy(image)

        # falls Daten nicht schon mit einer Dim mehr gespeichert !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! (Lymphome (Channels) / Patienten)
        #image = image.unsqueeze(0) # [1,512,512]
        #print(image.shape)

        # Falls nicht als float sondern als Torch Tensor abgespeichert:
        image = image.float() # Falls nicht schon als float gespeichert
        #print("float", image.shape)
        #print("lable", lable)

        return image, lable, name #+++++++++++++++++++++++++++++++



# Nur zum Testen:

if __name__ == '__main__':
    dataset = TorchDataset("/home/wolfda/Clinic_Data/Challenge/Challenge_COVID-19-20_v2/Train_tensor_slices_filter", augmentations=True)
    img, mask = dataset[1]
    img, mask = dataset[2]

    from batchviewer import view_batch

    view_batch(img, mask, width=512, height=512)
