# Self-Supervised Pre-Training with Contrastive and Masked Autoencoder Methods for Dealing with Small Datasets in Deep Learning for Medical Imaging

Paper accepted in Nature Scientific Reports. \
Nature DOI: https://doi.org/10.1038/s41598-023-46433-0 \
ArXiv Version: <https://arxiv.org/abs/2308.06534>

## Pre-Trained Models 
You can download the pre-trained model checkpoints here from Google Drive:


| Pre-Training  | Method                | Model       |Dowwnload Link |
| ------------- | -------------         |------------ | ------------  |
| BYOL          | Contrastive Learning  | ResNet50    |[BYOL_Checkpoint](https://drive.google.com/uc?export=download&id=1eBZYl1rXkKJxz42Wu75uzb1kLg8FTv1H)              |
| SwAV          | Contrastive Learning  | ResNet50    |[SwAV_Checkpoint](https://drive.google.com/uc?export=download&id=11OWRzifq_BXrcFMZ13H0HwS4UGcaiAn_)               |
| MoCoV2        | Contrastive Learning  | ResNet50    |[MoCoV2_Checkpoint](https://drive.google.com/uc?export=download&id=1hUr_6XdYxjB66ZYEGTqE7b8I88IN9a1l)            | 
| SaprK         | Masked Autoencoder    | ResNet50    |[SparK_Checkpoint](https://drive.google.com/uc?export=download&id=1kYFS67jH9s8kAmhNyf5wlRj_Gh9vTK_H)               |


Here is code to initialise a ResNet50 model from PyTorch with the pre-training weights stored in the Checkpoint:  
(pytorch==1.12.1 torchvision==0.13.1)

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
