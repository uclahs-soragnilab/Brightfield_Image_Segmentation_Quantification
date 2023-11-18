#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split images
"""
from pathlib import Path
import cv2 as cv
import numpy as np
import argparse
import segment_ml
import torch
from IPython import embed
from tqdm import tqdm
#from tkinter import filedialog as fd

#aux_params=dict(
#    pooling='avg',             # one of 'avg', 'max'
#    dropout=0.5,               # dropout ratio, default is None
#    activation='sigmoid',      # activation function, default is None
#    classes=2,                 # define number of output labels
#)

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def patch_images(imgpath, model):
    """
    Takes a full image tiff, break into 512 x 512 tiles, segment, and reassemble into full image mask
    Args:
        imgpath: path to image
        model_pkl: checkpoint model (.pkl file)
    Returns:
        None
    """
    img = cv.imread(str(imgpath), cv.IMREAD_GRAYSCALE)
    base_name = imgpath.name.strip('.tiff')
    # base_name = imgpath.name.strip('.png')
    base_path = str(imgpath.parent.resolve())
    # patch_directory = Path(f'{base_path}/patches/')
    ## initialize the directory if it doesn't exist
    #patch_directory.mkdir(parents=True, exist_ok=True)

    # add the difference of the remainder to expand the image
    new_x_dim = img.shape[0] + (512 - img.shape[0] % 512)
    new_y_dim = img.shape[1] + (512 - img.shape[1] % 512)
    # new_original = np.zeros((new_x_dim, new_y_dim))
    new_mask = np.zeros((new_x_dim, new_y_dim))
    # model = segment_ml.load_trained_model(model_pkl) # moved outside loop for speed
    for x_coor in tqdm(range(0,img.shape[0],512)):
        for y_coor in range(0,img.shape[1],512):
            tile = img[x_coor:x_coor+512, y_coor:y_coor+512]
            # make sure that the tile is 512 x 512, if not pad it
            if tile.shape != (512, 512):
                xpad = 512 - tile.shape[0]
                ypad = 512 - tile.shape[1]
                if xpad < 0:
                    xpad = 0
                if ypad < 0:
                    ypad = 0
                assert xpad >= 0
                assert ypad >= 0
                tile = np.pad(tile, ((0, xpad), (0, ypad)), mode = 'constant', constant_values = 0)
            # new_original[x_coor:x_coor+512, y_coor:y_coor+512] = tile
            mask_tile = segment_ml.predict_mask(tile, model)
            new_mask[x_coor:x_coor+512, y_coor:y_coor+512] = mask_tile
            # save patches
            # cv.imwrite(f"{str(patch_directory)}/{base_name}_{x_coor}_{y_coor}.png", tile)
            # cv.imwrite(f"{str(patch_directory)}/{base_name}_{x_coor}_{y_coor}_mask.png", np.array(mask_tile))
    #cv.imwrite(f"{base_path}/{base_name}_neworiginal.png", new_original)
    new_mask=new_mask[0:img.shape[0],0:img.shape[1]] # Removes added padding
    cv.imwrite(f"{base_path}/{base_name}_mask.png", new_mask)


def main():
    """
    Program takes a basepath, looks for all tiffs that are located inside,
    and then assembles the 512 x 512 images into a single png
    """
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--basepath', type=str, required=True)
    #args = parser.parse_args()

    #args = get_arguments()
    print("Processing tifs in the Image-Folder directory")
    #im_datapath = fd.askdirectory()
    im_datapath = "Image-Folder/"
    #print("Choose checkpoint pkl file for model")
    #checkpoint_pkl = fd.askopenfilename()
    checkpoint_pkl = "final_unet_checkpoint.pkl"

    model = segment_ml.load_trained_model(checkpoint_pkl)

    tiffs = list(Path(im_datapath).glob('**/*_1um.tiff'))
    for tif in tqdm(tiffs):
        patch_images(tif, model)


if __name__ == "__main__":
    main()
