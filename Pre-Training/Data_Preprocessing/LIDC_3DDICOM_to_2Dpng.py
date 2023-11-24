# 3D DICOM to 2D jpeg File for the LIDC data

import glob
import pydicom as dicom
import cv2
import numpy as np

# ============================== Image ===========================================

# ------------------- DICOM Image to Numpy Array (+Houndfield) ------------------------------
def get_pixels_hu(scans):

    image = scans.pixel_array  # CT Scan
    image = image.astype(np.int16) # to Numpy

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans.RescaleIntercept if 'RescaleIntercept' in scans else -1024
    slope = scans.RescaleSlope if 'RescaleSlope' in scans else 1

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)

# ------------------- Windowing --------------------------------------------
# von https://gist.github.com/lebedov/e81bd36f66ea1ab60a1ce890b07a6229
# abdomen: {'wl': 60, 'ww': 400} || angio: {'wl': 300, 'ww': 600} || bone: {'wl': 300, 'ww': 1500} || brain: {'wl': 40, 'ww': 80} || chest: {'wl': 40, 'ww': 400} || lungs: {'wl': -400, 'ww': 1500}
def win_scale(data, wl, ww, dtype, out_range):
    data_new = np.empty(data.shape, dtype=np.double)
    data_new.fill(out_range[1] - 1)

    data_new[data <= (wl - ww / 2.0)] = out_range[0]
    data_new[(data > (wl - ww / 2.0)) & (data <= (wl + ww / 2.0))] = \
        ((data[(data > (wl - ww / 2.0)) & (data <= (wl + ww / 2.0))] - (wl - 0.5)) / (ww - 1.0) + 0.5) * (
                out_range[1] - out_range[0]) + out_range[0]
    data_new[data > (wl + ww / 2.0)] = out_range[1] - 1

    return data_new.astype(dtype)

#  ------------------- Range (for png/jepg)-------------------------------------------
def interval_mapping(image, from_min, from_max, to_min, to_max):
    # map values from [from_min, from_max] to [to_min, to_max]
    # image: input array
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)


# =============================================================================
# Save
# =============================================================================

def save(image, save_path, patient_name, i, w1, w2):

    # --------------------------Image----------------------------------------------------
    # DICOM -> Numpy
    patient_dicom = dicom.dcmread(image)
    patient_pixels = get_pixels_hu(patient_dicom)  # -> Numpy Array

    # Windowing (We do not do a windowing since we want to use all data for pre-training) **
    #patient_pixels = win_scale(patient_pixels, w1, w2, type(patient_pixels), [patient_pixels.min(), patient_pixels.max()])  # Numpy Array Korrigiert

    # From hounsfield range to range [0,255] for jpeg/png
    patient_pixels = interval_mapping(patient_pixels, patient_pixels.min(), patient_pixels.max(), 0, 255)

    # Save
    path = save_path + "/" + str(patient_name) + "_" + str(i) + ".png"
    cv2.imwrite(path, patient_pixels)


# =============================================================================
# Main
# =============================================================================
def main():

    # ToDo: Path were the  DICOM files are saved:
    data_path = "/path/to/LIDC/manifest-1600709154662/LIDC-IDRI" # In this folder are many subfolders, one for each patient

    # ToDo: Path were the data should be saved
    save_path = "/folder/where/the/data/should/be/saved" # Create a folder on your computer where you want to save the pn images

    # ToDo: Choose a Window:
    #body_part = "abdomen": wl = 60, ww = 400
    #body_part == "angio": wl = 300, ww = 600
    #body_part == "bone": wl = 300, ww = 150
    #body_part == "brain": wl = 40, ww = 80
    #body_part == "chest": wl = 40, ww = 400
    #body_part == "lungs": wl = -400, ww = 1500
    wl = -400
    ww = 1500
    # We do not use this here, since we want to use all data for pre-training **


    i = 0

    # This loop saves one DICOM CT slice after another as png
    Ordner = sorted(glob.glob(data_path + "/*"))  # List: All paths from the LIDC-IDRI folder (folder of the individual patients)
    for fileA in Ordner:  # Runs through all paths in the LIDC-IDRI folder (all patients)
        patient_name = fileA.split("/")[-1]  # Name of the patient

        Ordner2 = sorted(glob.glob(fileA + "/*")) # List: All paths from one patient folder (one X-ray, one CT)
        for fileB in Ordner2: # Runs through all paths in the patient folder (X-ray and CT)

            Ordner3 = sorted(glob.glob(fileB + "/*"))  #  Unnecessary subfolder that comes with the download
            for fileC in Ordner3:

                Ordner4 = sorted(glob.glob(fileC + "/*"))  # Paths of all DICOM files (either CT or X-ray)
                # Check whether CT or X-ray:
                number_files = len(Ordner4) # Number of files in the folder
                if number_files > 10: # Only if more than 10 files in the folder == CT (as X-ray only a few files)
                    for fileD in Ordner4: # Runs through all DICOM files

                        # Check whether dicom or xml
                        if fileD.endswith(".dcm"):

                            # (path of the series (DICOM files), patient name, shift number, WL, WW)
                            save(fileD, save_path, patient_name, i, wl, ww)

                            i = i+1


if __name__ == '__main__':
    main()