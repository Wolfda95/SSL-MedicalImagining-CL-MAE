from logging import getLogger
import torchvision.datasets as datasets


logger = getLogger()


class MoCoDataset(datasets.ImageFolder):
    def __init__(
        self,
        data_path,
        tau_g,
    ):
        super(MoCoDataset, self).__init__(data_path)
        self.tau_g = tau_g  # set of global transforms

    def __len__(self):
        return len(self.samples)

    def __str__(self):
        return f"LoGoDataset with {self.__len__()} images"

    def __getitem__(self, idx):
        path, _ = self.samples[idx]
        image = self.loader(path)

        global_crops = list(map(lambda transform: transform(image), self.tau_g))

        return (global_crops[0], global_crops[1]), 0
