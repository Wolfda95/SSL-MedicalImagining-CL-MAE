# Pre-Training with SwAV, MoCoV2, BYOL

We used the implementation of PyTorch Lightning Bolds [https://lightning.ai/docs/pytorch/stable/ecosystem/bolts.html](https://lightning.ai/docs/pytorch/stable/ecosystem/bolts.html)


1. Download the LIDC data and run the preprocessing script as explained here: [https://github.com/Wolfda95/SSL-MedicalImagining-CL-MAE/tree/main/Pre-Training/Data_Preprocessing](https://github.com/Wolfda95/SSL-MedicalImagining-CL-MAE/tree/main/Pre-Training/Data_Preprocessing)
2. Change the folder structure of the preprocessed data to: (Take part of the images as validation) 
    ```bash
        LIDC-Data
         /        
       train       
    ```
2. Open your terminal and follow these steps: 
    1. <code>conda create --name SSL_Downstream python==3.8</code>
    2. <code>conda activate SSL_Downstream</code>
    3. <code>conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch</code>
    4. <code>cd .../SSL-MedicalImagining-CL-MAE/Pre-Training/Masked_Autoencoder/</code>
    5. <code>pip install -r requirements.txt</code>
4. Start the pre-training with a bash script:
    ```bash
    #!/bin/bash
    
    python ./main.py \
    --exp_name=ResNet50_1 \
    --data_path=/path/to/LIDC-Data \
    --model=resnet50 \
    --bs=32 \
    --exp_dir=/path/to/where/results/should/be/saved \
    --ep=1600 \
    ```
For further information and other setting please refere to the SparK github: [https://github.com/keyu-tian/SparK](https://github.com/keyu-tian/SparK4)
