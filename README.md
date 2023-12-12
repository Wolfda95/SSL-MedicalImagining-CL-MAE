# Self-Supervised Pre-Training with Contrastive and Masked Autoencoder Methods for Dealing with Small Datasets in Deep Learning for Medical Imaging

Publication about self-supervised pre-training in medical imaging accepted in Nature Scientific Reports. \
Nature: https://doi.org/10.1038/s41598-023-46433-0 \
ArXiv: <https://arxiv.org/abs/2308.06534>

## Introduction
Training deep learning models requires large datasets with annotations for all training samples. However, in the medical imaging domain, annotated datasets for specific tasks are often small due to the high complexity of annotations, limited access, or the rarity of diseases. To address this challenge, deep learning models can be pre-trained on large image datasets without annotations using methods from the field of self-supervised learning.
In this paper we compare state-of-the-art self-supervised pre-training methods based on contrastive learning ([SwAV](https://proceedings.neurips.cc/paper/2020/hash/70feb62b69f16e0238f741fab228fec2-Abstract.html), [MoCo](https://openaccess.thecvf.com/content_CVPR_2020/html/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.html), [BYOL](https://proceedings.neurips.cc/paper_files/paper/2020/file/f3ada80d5c4ee70142b17b8192b2958e-Paper.pdf)) and masked autoencoders ([SparK](https://openreview.net/forum?id=NRxydtWup1S)) for convolutional neural networks (CNNs).

![SSL](https://github.com/Wolfda95/SSL-MedicalImagining-CL-MAE/assets/75016933/cf1589b7-4ea7-463e-866b-15586e131cd0)

Due to the challenge of obtaining sufficient annotated training data in medical imaging, it is of particular interest to evaluate how the self-supervised pre-training methods perform when fine-tuning on small datasets. Our experiments show, that the SparK pre-training method is more robust to the training downstream dataset size than the contrastive methods. Based on our results, we propose the SparK pre-training for medical imaging tasks with only small annotated datasets.

## Code 

### 1) Pre-Training
First, the deep learning model needs to be pre-trained with a large dataset of images without annotations. \
Go to the folder [Pre-Training](https://github.com/Wolfda95/SSL-MedicalImagining-CL-MAE/tree/main/Pre-Training) for the the pre-training code and further explanations.. \
You can download our pre-trained models below.

### 2) Downstream
The pre-training is evaluated on three downstream classification tasks. \
You can test the downstream tasks with the pre-trained models you can download below. \
Go to the folder [Downstream](https://github.com/Wolfda95/SSL-MedicalImagining-CL-MAE/tree/main/Downstream) for the the downstream code and further explanations.

## Pre-Trained Models 
You can download the pre-trained model checkpoints here from Google Drive:


| Pre-Training  | Method                | Model       |Dowwnload Link |
| ------------- | -------------         |------------ | ------------  |
| BYOL          | Contrastive Learning  | ResNet50    |[BYOL_Checkpoint](https://drive.google.com/uc?export=download&id=1eBZYl1rXkKJxz42Wu75uzb1kLg8FTv1H)              |
| SwAV          | Contrastive Learning  | ResNet50    |[SwAV_Checkpoint](https://drive.google.com/uc?export=download&id=11OWRzifq_BXrcFMZ13H0HwS4UGcaiAn_)               |
| MoCoV2        | Contrastive Learning  | ResNet50    |[MoCoV2_Checkpoint](https://drive.google.com/uc?export=download&id=1hUr_6XdYxjB66ZYEGTqE7b8I88IN9a1l)            | 
| SaprK         | Masked Autoencoder    | ResNet50    |[SparK_Checkpoint](https://drive.google.com/uc?export=download&id=1kYFS67jH9s8kAmhNyf5wlRj_Gh9vTK_H)               |


Here is code to initialise a ResNet50 model from PyTorch with the pre-training weights stored in the Checkpoint:  \
(pytorch==1.12.1 torchvision==0.13.1) \
You can also check out the the [Downstream](https://github.com/Wolfda95/SSL-MedicalImagining-CL-MAE/tree/main/Downstream) code where this is already implemented.

```python

# Fill out: 
# Choose the Pre-Training Method here [options: "SparK", "SwAV", "MoCo", "BYOL"]
pre_train = "SparK"
# Insert the downloaded file hier (.ckpt or .pth) 
pre_training_checkpoint = "/path/to/download/model.ckpt"

# PyTorch Resnet Model
res_model = torchvision.models.resnet50()

# Load pre-training weights
state_dict = torch.load(pre_training_checkpoint)

# Match the correct name of the layers between pre-trained model and PyTorch ResNet
# Extraction:
if "module" in state_dict: # (SparK)
    state_dict = state_dict["module"] 
if "state_dict" in state_dict: # (SwAV, MoCo, BYOL) 
    state_dict = state_dict["state_dict"]
# Replacement: 
if pre_train == "SparK" or pre_train == "SwAV":
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}  
elif pre_train == "MoCo":
    state_dict = {k.replace("encoder_q.", ""): v for k, v in state_dict.items()} 
elif pre_train == "BYOL":
    state_dict = {k.replace("online_network.encoder.", ""): v for k, v in state_dict.items()}

# Initialisation of the ResNet model with pre-training checkpoints
pretrained_model = res_model.load_state_dict(state_dict, strict=False)

# Check if it works
print(format(pretrained_model))
# If this appears, everything is correct: 
# missing_keys=
  # ['fc.weight', 'fc.bias'] (beacuse the last fully connected layer was not pre-trained) 
# unexpected_keys= 
  # MoCo: All "encoder_k" layers (because MoCo has 2 encoders and we use only encoder_q)
  # BYOL: All "online_network.projector" and "target_network.encoder" layers (because BYOL has 2 encoders and we only the online_network.encoder)
  # SwAV: All "projection_head" layers (beacuse SwAV has an aditional projection head for the online clustering) 
  # SparK: []

```

## Contact
This work was done in a collaboration between the [Clinic of Radiology](https://www.uniklinik-ulm.de/radiologie-diagnostische-und-interventionelle.html) and the [Visual Computing Research Group](https://viscom.uni-ulm.de/) at the Univerity of Ulm.

My Profiles: 
- [Ulm University Profile](https://viscom.uni-ulm.de/members/daniel-wolf/)
- [Personal Website](https://wolfda95.github.io/)
- [Google Scholar](https://scholar.google.de/citations?hl=de&user=vqKsXwgAAAAJ)
- [Orcid](https://orcid.org/0000-0002-8584-5189)
- [LinkedIn](https://www.linkedin.com/in/wolf-daniel/)

If you have any questions, please email me:
[daniel.wolf@uni-ulm.de](mailto:daniel.wolf@uni-ulm.de)

## Cite
```latex
@article{wolf2023self,
  title={Self-supervised pre-training with contrastive and masked autoencoder methods for dealing with small datasets in deep learning for medical imaging},
  author={Wolf, Daniel and Payer, Tristan and Lisson, Catharina Silvia and Lisson, Christoph Gerhard and Beer, Meinrad and G{\"o}tz, Michael and Ropinski, Timo},
  journal={Scientific Reports},
  volume={13},
  number={1},
  pages={20260},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```
