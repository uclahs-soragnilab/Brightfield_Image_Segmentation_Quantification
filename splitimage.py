#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split images
"""
#from image_slicer import slice as sliceimage
from tkinter import filedialog as fd
from pathlib import Path
import cv2 as cv
import numpy as np


def slice_images(imgpath):
    """
    Takes an image path and breaks it into 512 x 512 tiles
    Args:
        imgpath: path to image
    Returns:
        None
    """
    img = cv.imread(str(imgpath), cv.IMREAD_GRAYSCALE)
    base_name = str(imgpath.resolve()).strip('.tiff')
    for x_coor in range(0,img.shape[0],512):
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
            cv.imwrite(f"{base_name}_{x_coor}_{y_coor}.png", tile)

def main():
    """
    Program takes a basepath, looks for all tiffs that are located inside,
    and then breaks each tiff into a 512 x 512 png
    """
    
    foldername = fd.askdirectory()

    tiffs = list(Path(foldername).glob('**/*.tiff'))
    for tif in tiffs:
        slice_images(tif)


if __name__ == "__main__":
    main()
