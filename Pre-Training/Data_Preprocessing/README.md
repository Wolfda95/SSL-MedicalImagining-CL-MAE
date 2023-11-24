# Prprocessing

Download the LIDC-IDRI Dataset from here: [https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254) 

The data comes as DICOM images. We save each slice of each CT image as a png file. We do not applay andy windwoing since we want to use all data for pre-training. 

### Start: 
If you are using Conda on Linux, here is how to get started: 
1. Open your terminal and follow these steps: 
    1. <code>conda create --name SSL_Preprocessing python==3.10</code>
    2. <code>conda activate SSL_Downstream</code>
    4. <code>cd ...SSL-MedicalImagining-CL-MAE/Pre-Training/Data_Preprocessing</code>
    5. <code>pip install -r requirements.txt</code>
2. Open LIDC_3DDICOM_to_2Dpng.py and adjust the folder pathes in the main method

