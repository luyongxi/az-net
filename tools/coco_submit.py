#!/usr/bin/env python

# --------------------------------------------------------
# Object detection using AZ-Net
# Written by Yongxi Lu
# Modified from Fast R-CNN
# --------------------------------------------------------

"""Convert detections result into COCO submission file(s)"""

import _init_paths
import argparse
import sys, os
import json
import cPickle
import datasets.coco

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Convert results in to COCO submission file')
    parser.add_argument('--size', dest='max_size', help='Max size of each file (in number of images)',
                        default=30000, type=int)
    parser.add_argument('--file', dest='filename',
                    help='filename for detection file',
                    default='detections.pkl', type=str)
    parser.add_argument('--split', dest='split',
                        help='split of the dataset',
                        default='test', type=str)
    parser.add_argument('--year', dest='year',
                        help='year of the dataset',
                        default='2015',type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)
    
    with open(args.filename, 'rb') as f:
        all_boxes = cPickle.load(f)
    
    output_dir = os.path.dirname(args.filename)
    coco_dataset = datasets.coco(args.split, args.year)
    
    coco_dataset.write_coco_multiple_files(all_boxes, args.max_size, output_dir)
        
