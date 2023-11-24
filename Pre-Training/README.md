# Pre-Training

## 1) Data Preprocessing
We use the [LIDC-IDRI](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254)  dataset for self-supervised pre-training. Only the CT images are used without any labels or other information. \
We perform the pre-training on the 2D slices of the CT volumes. \
Go to the folder [Data_Preprocessing](https://github.com/Wolfda95/SSL-MedicalImagining-CL-MAE/tree/main/Pre-Training/Data_Preprocessing) for the preprocessing code.

## 2) Pre-Training
We compare two types of self-supervised pre-training: Contrastive Learning and Masked Autoencoder. 

### a) Contrastive Learning
Three state-of-the-art and best-performing contrastive learning methods on convolutional networks are: 
[SwAV](https://proceedings.neurips.cc/paper/2020/hash/70feb62b69f16e0238f741fab228fec2-Abstract.html), [MoCo](https://openaccess.thecvf.com/content_CVPR_2020/html/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.html), and [BYOL](https://proceedings.neurips.cc/paper_files/paper/2020/file/f3ada80d5c4ee70142b17b8192b2958e-Paper.pdf) 

Go to the folder [Contrastive_Learning](https://github.com/Wolfda95/SSL-MedicalImagining-CL-MAE/tree/main/Pre-Training/Contrastive_Learning) for the Contrastive Learning code.

### b) Masked Autoencoder
In a recent study published at the eleventh International Conference on Learning Representations 2023, Tian et al. demonstrate that Masked Autoencoder can be adapted for convolutional models using sparse convolutions. Their new approach, called [SparK](https://proceedings.neurips.cc/paper/2020/hash/70feb62b69f16e0238f741fab228fec2-Abstract.html), outperforms all state-of-the-art contrastive methods on a convolutional model, using natural images from ImageNet for self-supervised pre-training. We apply and investigate the Spark pre-training method to CT images. 

Go to the folder [Masked_Autoencoder](https://github.com/Wolfda95/SSL-MedicalImagining-CL-MAE/tree/main/Pre-Training/Masked_Autoencoder) for the SparK code.
