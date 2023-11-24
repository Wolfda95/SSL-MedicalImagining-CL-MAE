from pl_bolts.models.gans.basic.basic_gan_module import GAN
from pl_bolts.models.gans.dcgan.dcgan_module import DCGAN
from pl_bolts.models.gans.pix2pix.pix2pix_module import Pix2Pix
from pl_bolts.models.gans.srgan.srgan_module import SRGAN
from pl_bolts.models.gans.srgan.srresnet_module import SRResNet

__all__ = [
    "GAN",
    "DCGAN",
    "Pix2Pix",
    "SRGAN",
    "SRResNet",
]
