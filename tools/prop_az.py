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

"""Use AZ-net to generate object proposals on an image database."""

import _init_paths
from detect.test import test_proposals
from detect.config import cfg, cfg_from_file
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time, os, sys

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Use AZ-Net to generate proposals')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the SC-Net network',
                        default=None, type=str)
    parser.add_argument('--def_fc', dest='prototxt_fc',
                        help='prototxt file defining the fully-connected layers of SC-Net network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='AZ-Net model to test',
                        default=None, type=str)                     
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    while not os.path.exists(args.caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel))
        time.sleep(10)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    
    # full SC-net
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]
    # fc layers of SC-Net
    net_fc = caffe.Net(args.prototxt_fc, args.caffemodel, caffe.TEST)
    net_fc.name = os.path.splitext(os.path.basename(args.caffemodel))[0]
    
    nets = {'full':net, 'fc': net_fc}
    
    imdb = get_imdb(args.imdb_name)

    test_proposals(nets, imdb)
