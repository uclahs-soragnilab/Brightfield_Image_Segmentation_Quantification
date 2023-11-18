#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs pipeline from creating masks from tiffs to calculating area for each sample
"""
import sys
import argparse
sys.path.append('cell_segmentation_ML/')
import mask_whole_image as mwi
sys.path.append('cell_segmentation_ML/utilities/')
import calculate_area as calc_a

def get_arguments():
    """
    Read command line arguments
    """
    parser = argparse.ArgumentParser(description=" Run Segmentation Pipeline Arguments ")
    parser.add_argument(
        "--mask",
        action='store_true',
        help="execute script in segmentation mode to generate a mask png image",
    )
    parser.add_argument(
        "--quantify",
        action='store_true',
        help="execute script in quantification mode to generate a summary csv for each sample",
    )
    args = parser.parse_args()
    return args


def main():
    """
    Program takes a basepath, looks for all tiffs that are located inside,
    and then assembles the 512 x 512 images into a single png
    """
    args = get_arguments()
    if args.mask:
        mwi.main()
    if args.quantify:
        calc_a.main()


if __name__ == "__main__":
    main()