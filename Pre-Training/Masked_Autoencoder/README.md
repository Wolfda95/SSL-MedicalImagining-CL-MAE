# Pre-Training with SparK

SparK is the first succesfull adaption of Masked Autoencoder Self-Supervised Pre-Training on Convolutional Neural Networks (CNNs).

This is code from the official implementation of SparK [https://github.com/keyu-tian/SparK](https://github.com/keyu-tian/SparK4) (MIT license)

### How to Start: 
You need to download the LIDC Data and run the preprocessing script 
If you are using Conda on Linux, here is how to get started: 
1. Open your terminal and follow these steps: 
    1. <code>conda create --name SSL_Downstream python==3.8</code>
    2. <code>conda activate SSL_Downstream</code>
    3. <code>conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch</code>\
    4. <code>cd .../SSL-MedicalImagining-CL-MAE/Pre-Training/Masked_Autoencoder/</code>
    5. <code>pip install -r requirements.txt</code>
4. Start the pre-training with a bash script:
    ```bash
    #!/bin/bash

    cd .../SSL-MedicalImagining-CL-MAE/Pre-Training/Masked_Autoencoder/
    python3 main.py --exp_name=ResNet50_1 --data_path=/path/to/LIDC/Data --model=resnet50 --bs=32
    ```
For further information and other setting please refere to the SparK github: [https://github.com/keyu-tian/SparK](https://github.com/keyu-tian/SparK4)


### SparK Paper
Please also cite the SparK paper: 

```latex
@inproceedings{
tian2023designing,
title={Designing {BERT} for Convolutional Networks: Sparse and Hierarchical Masked Modeling},
author={Keyu Tian and Yi Jiang and qishuai diao and Chen Lin and Liwei Wang and Zehuan Yuan},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=NRxydtWup1S}
}
```
