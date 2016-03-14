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

"""Train a Fast R-CNN network using proposals from AZ-Net."""

import _init_paths
from detect.train_det import get_training_roidb, train_net
from detect.config import cfg, cfg_from_file, get_output_dir, cfg_set_mode
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import numpy as np
import sys, os

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--norm', dest='normalize',
                        help='to un-normalize (use when pre-trained model is normalized)',
                        action='store_true')
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the AZ-Net network',
                        default=None, type=str)
    parser.add_argument('--def_fc', dest='prototxt_fc',
                        help='prototxt file defining the fully-connected layers of AZ-Net network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='AZ-Net model to test',
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

    cfg_set_mode('Train')

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)
        
    if args.normalize:
        cfg.TRAIN.UN_NORMALIZE = True
    else:
        cfg.TRAIN.UN_NORMALIZE = False

    # set up caffe
    caffe.set_mode_gpu()
    if args.gpu_id is not None:
        caffe.set_device(args.gpu_id)
        
        # set up test net, if provided
    # full AZ-net
    nets = None
    if args.caffemodel is not None:
        net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
        net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]
        # pooling layers of SC-Net
        net_fc = caffe.Net(args.prototxt_fc, args.caffemodel, caffe.TEST)
        net_fc.name = os.path.splitext(os.path.basename(args.caffemodel))[0]
        nets = {'full':net, 'fc': net_fc}


    imdb = get_imdb(args.imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    roidb = get_training_roidb(imdb, nets)
    
    del nets, net, net_fc

    output_dir = get_output_dir(imdb, None)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    train_net(args.solver, roidb, output_dir,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters)
