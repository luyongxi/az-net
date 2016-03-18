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

"""Use AZ-Net and Fast R-CNN to perform object detection w/ shared convolutional layers"""

import _init_paths
from detect.test import test_net_shared
from detect.config import cfg, cfg_from_file, cfg_set_mode, cfg_load_thresh, cfg_set_path
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time, os, sys

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Use Fast-RCNN for object detection')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def_fc_frcnn', dest='prototxt_fc_frcnn',
                        help='prototxt file defining the Fast-RCNN network',
                        default=None, type=str)
    parser.add_argument('--net_frcnn', dest='caffemodel_frcnn',
                        help='Fast R-CNN model to test',
                        default=None, type=str)
    parser.add_argument('--def_az', dest='prototxt_az',
                        help='prototxt file defining the AZ-Net network',
                        default=None, type=str)
    parser.add_argument('--def_fc_az', dest='prototxt_fc_az',
                        help='prototxt file defining the fully-connected layers of AZ-Net network',
                        default=None, type=str)
    parser.add_argument('--net_az', dest='caffemodel_az',
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
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--thresh', dest='thresh_file',
                        help='file that stores zoom threshold',
                        default=None, type=str)
    parser.add_argument('--exp', dest='exp_dir',
                        help='experiment path',
                        default=None, type=str)

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

    cfg_set_path(args.exp_dir)

    while not os.path.exists(args.thresh_file) and args.wait:
        print('Wating for {} to exist...'.format(args.thresh_file))
        time.sleep(10)

    thresh = cfg_load_thresh(args.thresh_file)
    cfg_set_mode('Test', thresh)

    print('Using config:')
    pprint.pprint(cfg)

    while not os.path.exists(args.caffemodel_az) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel_az))
        time.sleep(10)
        
    while not os.path.exists(args.caffemodel_frcnn) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel_frcnn))
        time.sleep(10)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    
    # AZ-Net
    az_net = caffe.Net(args.prototxt_az, args.caffemodel_az, caffe.TEST)
    az_net.name = os.path.splitext(os.path.basename(args.caffemodel_az))[0]
    az_net_fc = caffe.Net(args.prototxt_fc_az, args.caffemodel_az, caffe.TEST)
    az_net_fc.name = os.path.splitext(os.path.basename(args.caffemodel_az))[0]
    az_nets = {'full':az_net, 'fc': az_net_fc}
    
    # Fast R-CNN network
    frcnn_net_fc = caffe.Net(args.prototxt_fc_frcnn, args.caffemodel_frcnn, caffe.TEST)
    frcnn_net_fc.name = os.path.splitext(os.path.basename(args.caffemodel_frcnn))[0]
    frcnn_nets = {'fc': frcnn_net_fc}
    
    imdb = get_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)

    test_net_shared(az_nets, frcnn_nets, imdb)
