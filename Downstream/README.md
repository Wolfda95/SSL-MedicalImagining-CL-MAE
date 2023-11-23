# Downstream Tasks

We tested our pre-training on three CT classification tasks: 
- **COVID-19**: Covid Classification on Lung CT scans (From Grand Challenge [https://covid-ct.grand-challenge.org/](https://covid-ct.grand-challenge.org/) or 
[https://doi.org/10.48550/arXiv.2003.13865](https://doi.org/10.48550/arXiv.2003.13865))
- **OrgMNIST**: Multi class classification of 11 body organs on patches cropped around organs of abdominal CT scans (From MedMNIST Challenges [https://medmnist.com/](https://medmnist.com/) or [https://doi.org/10.1038/s41597-022-01721-8](https://doi.org/10.1038/s41597-022-01721-8)) 
- **Brain**: Brain hemmorage classification on brain CT scans on an internal dataset of the Ulm Univerity Medical Center

  
Here are the jupyther notebooks for the three Downstream Tasks.

### How to Start: 
If you use Conda on Linux, here is how to start: 
1. open your Terminal and do the following steps: 
    1. <code>conda create --name SSL_Downstream python==3.10</code>
    2. <code>conda activate SSL_Downstream</code>
    3. *CUDA 10.2:* <code>conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch</code>\
       *CUDA 11.3:* <code>conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch</code>\
       *CUDA 11.6:* <code>conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forgey</code>
    4. <code>cd ...SSL-MedicalImagining-CL-MAE/Downstream/</code>
    5. <code>pip install -r requirements.txt</code>
2. Download JupyterLab
3. Login to Wandb (or create an account [https://wandb.ai/](https://wandb.ai/))
4. Open the Notebooks "OrgMNIST.ipynb" or "COVID_19.ipynb" or "Brain.ipynb"
    1. Fill out the first cell with your preferences (Here you have to add the path to the downloaded pre-training checkpoints from the main README.md)
    2. Run all cells 
