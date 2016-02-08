#!/usr/bin/env python

# --------------------------------------------------------
# Object detection using AZ-Net
# Written by Yongxi Lu
# Modified from Fast R-CNN
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compute the pixel means of a given dataset"""

import _init_paths
from datasets.factory import get_imdb
import numpy as np
import argparse
import sys
import cv2

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Compute pixel means of imdb')
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_trainval', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)
    
    imdb = get_imdb(args.imdb_name)
    num_images = len(imdb.image_index)
    
    # means of pixel values, in BGR order
    means = np.zeros((3,))
    num_pixels = 0.0
    
    for i in xrange(num_images):
        im = cv2.imread(imdb.image_path_at(i)) 
        im_means = im.mean(axis=(0,1))
        im_num_pixels = float(im.shape[0] * im.shape[1])
        means = means * num_pixels / (num_pixels + im_num_pixels) \
            + im_means * im_num_pixels / (num_pixels + im_num_pixels)
        num_pixels = num_pixels + im_num_pixels
        
        if i % 1000 == 0 or i == num_images-1:
            print 'Processing {}/{}, the mean is ({})'.format(i, num_images,means)
