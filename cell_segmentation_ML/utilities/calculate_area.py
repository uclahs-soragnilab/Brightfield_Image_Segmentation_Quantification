#!/usr/bin/env python3
"""
The purpose of this program is to take an cell image mask
and return the centroid positions and area of the masks
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from skimage import filters
from skimage import segmentation, data, io
from PIL import Image
import numpy
from scipy.ndimage import label
import math
import argparse
import scipy.io
#from tkinter import filedialog as fd
from IPython import embed
from tqdm import tqdm

def load_image(imgpath):
    """
    Args:
        imgpath - Path to image
    Returns:
        img - numpy.ndarray
    """
    img = cv.imread(str(imgpath), cv.IMREAD_GRAYSCALE)
    # binarize the image, and set the masks to have value
    img = np.where(img > 100, 0, 255)
    img = img.astype('uint8')
    return(img)

def cell_finder(img):
    """
    Find the cells
    Args:
        img - as read by load_image
    Returns:
        data - numpy array with xpos_centroid, ypos_centroid, size of cell
    """
    nb_components, output, stats, centroids = \
        cv.connectedComponentsWithStats(img, connectivity=8)
    sizes = stats[1:, -1]
    locations = centroids[1:, ]
    sizes = sizes.reshape((-1, 1))
    # form is xpos, ypos, size
    data = np.append(locations, sizes, axis = 1)
    data = pd.DataFrame(data, columns = ['x_centroid', 'y_centroid', 'area'])
    return data

def summarize_csv(parent_dir):
    """
    Iterate over folder containing csvs with connected regions data, 
    calclate total area, # of regions, append to table in parent dir
    Args:
        parent_dir - dir containing directories of csvs
    Returns:
        none
    """

    csv_list = sorted(Path(parent_dir).glob('**/*_mask.csv'))
    #To be able to run this script on both a parent folder with multiple samples in the subfolders 
    #We find the CSV files then go back 2 subdirectories, one for the day and one for all days of the sample
    #Then loop in each folder including all exported scan days of an individual sample

    sample_parent_folders = []
    loop_list = len(csv_list)
    for item in range(loop_list):
        d = csv_list[item]
        folders_of_days = d.parent
        folder_of_sample = folders_of_days.parent
        sample_parent_folders.append(folder_of_sample)

    distinct_folders = set(sample_parent_folders)
    distinct_folders = list(distinct_folders)

    
    for sample_folder in distinct_folders:
        csv_list = sorted(Path(sample_folder).glob('**/*_mask.csv'))
        col_names = ['sample','day','well','channel','total_number_regions','total_area']
        summary_data = pd.DataFrame(columns=col_names)

        for csv_file in tqdm(csv_list):
            csv_tmp = pd.read_csv(csv_file, sep=',')

            num_regions = np.shape(csv_tmp)[0] - 1
            total_area = csv_tmp.sum()[2]

            # Parse file info for csv table
            sample_name = re.search('[^/]+$', str(sample_folder).replace('\\', '/')).group(0) # replace('\\', '/') is because the paths were written as "\\"
            day_name = re.search('(?<=_)([^_]*)(?=\/Well)',str(csv_file).replace('\\', '/')).group(0) # replace('\\', '/') is because paths were written as "\\"
            well_name = re.search('(?<=/Well_)(.+)(?=_Ch)',str(csv_file).replace('\\', '/')).group(0) # replace('\\', '/') is because paths were written as "\\"
            channel_name = re.search(f'(?<=/Well_{well_name}_)(Ch[0-9])(?=_[0-9]um_mask.csv)',str(csv_file).replace('\\', '/')).group(0) # replace('\\', '/') is because paths were written as "\\"

            tmp_summary_data = pd.DataFrame([[sample_name,day_name,well_name,channel_name,num_regions,total_area]],columns=col_names)
            summary_data = pd.concat([summary_data,tmp_summary_data])
        # write summary_data to csv file in parent dir
        summary_data.to_csv(str(sample_folder)+f"/{sample_name}_all_days_summary.csv", index=False)


def main():
    #print("Choose directory with the mask pngs")
    #images_directory = fd.askdirectory()
    images_directory = "Image-Folder/"

    images = sorted(Path(images_directory).glob('**/*_mask.png'))
    for file in tqdm(images):
        fname = file.resolve()
        results = cell_finder(load_image(file))
        oname = str(fname).strip(".png") + ".csv"
        results.to_csv(oname, sep = ',', index = False)

    # Save summary csv with data from all subfolders aggregated
    summarize_csv(images_directory)

if __name__ == "__main__":
    main()
