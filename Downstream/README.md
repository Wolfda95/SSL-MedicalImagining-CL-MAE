# Downstream Tasks

We tested our pre-training on three CT classification tasks: 
- **COVID-19**: Covid classification on lung CT scans (From Grand Challenge [https://covid-ct.grand-challenge.org/](https://covid-ct.grand-challenge.org/) or 
[https://doi.org/10.48550/arXiv.2003.13865](https://doi.org/10.48550/arXiv.2003.13865))
- **OrgMNIST**: Multi-class classification of 11 body organs on patches cropped around organs from abdominal CT scans (From MedMNIST Challenges [https://medmnist.com/](https://medmnist.com/) or [https://doi.org/10.1038/s41597-022-01721-8](https://doi.org/10.1038/s41597-022-01721-8)) 
- **Brain**: Brain hemorrhage classification on brain CT scans on an internal dataset of the Ulm Univerity Medical Center

We gradually reduced the training dataset size for all three tasks to evaluate which pre-training method is best when only small annotated datasets are available. 

Here are our results: 
![Results](https://github.com/Wolfda95/SSL-MedicalImagining-CL-MAE/assets/75016933/83df9ede-bbf9-4eea-816c-f9de718ee764)



### How to Start: 
We have jupyther notebooks with PyTorch Lightning and Moani for the three Downstream Tasks. \
If you are using Conda on Linux, here is how to get started: 
1. Open your terminal and follow these steps: 
    1. <code>conda create --name SSL_Downstream python==3.10</code>
    2. <code>conda activate SSL_Downstream</code>
    3. *CUDA 10.2:* <code>conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch</code>\
       *CUDA 11.3:* <code>conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch</code>\
       *CUDA 11.6:* <code>conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge</code> \
       (The newest PyTorch should also work [https://pytorch.org/](https://pytorch.org/))
    4. <code>cd ...SSL-MedicalImagining-CL-MAE/Downstream/</code>
    5. <code>pip install -r requirements.txt</code>
2. Download Jupyter
3. Login to Wandb (or create an account [https://wandb.ai/](https://wandb.ai/))
4. Open "OrgMNIST.ipynb" or "COVID_19.ipynb" or "Brain.ipynb" in Jupyter Notebook or Jupyter Lab
    1. Fill out the first cell with your preferences (Here you have to add the path to the downloaded pre-training checkpoints from the main README.md)
    2. Run all cells 


### Start Notebooks from Bash:
This is not necessary, you can run everything directly in Jupyter Notebook or Jupyter Lab. However this might be useful
1. Open the notebook in Jupyter Lab
2. Click in the first code cell (This cell has all the parameters that needs to be specified)
    1. On the left click on the two gear wheels
    2. Add a cell tag with the name "parameters" \
     ![Parameters](https://github.com/Wolfda95/SSL-MedicalImagining-CL-MAE/assets/75016933/afcd9342-a6a7-4921-a25a-c1fdcc827cd6)
3. Download papermill <code>conda install -c conda-forge papermill</code>
4. Creat a bash file (e.g. "file.sh"). All variables from the first code cell are parameters and can be specified in the bash file with -p ...
   
```bash
# COVID-19
papermill COVID-19.ipynb COVID-19.ipynb \
-p root_dir "path/where/results/should/be/saved" \
-p Run "WandB_Name_of_Run" \
-p pretrained_weights "/path/to/the/downloaded/checkpoints/SparK.pth" \
-p pre_train "SparK" \

# OrgMNIST
papermill OrgMNIST.ipynb OrgMNIST.ipynb \
-p root_dir "path/where/results/should/be/saved" \
-p Run "WandB_Name_of_Run" \
-p pretrained_weights "/path/to/the/downloaded/SwAV.ckpt" \
-p pre_train "SwAV" \

```
5. Run the bash file (this will start the notebook)


   
