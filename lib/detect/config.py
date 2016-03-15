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

"""AZ-detect config system. (Modified from the Fast R-CNN config system)

This file specifies default config options for Fast R-CNN detector and AZ-Net. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.

Most tools in $ROOT/tools take a --cfg option to specify an override file.
    - See tools/{train,test}_net.py for example code that uses cfg_from_file()
    - See experiments/cfgs/*.yml for example YAML config override files
"""

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# Training options
#

__C.TRAIN = edict()

# Scales to use during training (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TRAIN.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
__C.TRAIN.MAX_SIZE = 1000

# Images to use per minibatch
__C.TRAIN.IMS_PER_BATCH = 2

# Minibatch size (number of regions of interest [ROIs])
__C.TRAIN.BATCH_SIZE = 128

# Fraction of minibatch that is labeled foreground (i.e. class > 0)
__C.TRAIN.FG_FRACTION = 0.25

# Fraction of minibatch that has positive labels in az-net
__C.TRAIN.AZ_POS_FRACTION = 0.5

# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.5

# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.1

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True

# Train bounding-box regressors
__C.TRAIN.BBOX_REG = True

# Overlap required between a ROI and ground-truth box in order for that ROI to
# be used as a bounding-box regression training example
__C.TRAIN.BBOX_THRESH = 0.5

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 10000

# Use caching region proposals or not
__C.TRAIN.USE_CACHE = False

# solver.prototxt specifies the snapshot path prefix, this adds an optional
# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
__C.TRAIN.SNAPSHOT_INFIX = ''

# Use a prefetch thread in roi_data_layer.layer
# So far I haven't found this useful; likely more engineering work is required
__C.TRAIN.USE_PREFETCH = False

__C.TRAIN.ADDREGIONS =[[0, 0, 1, 1],
                      [0, 0, 0.8, 0.8],
                      [0, 0.2, 0.8, 1],
                      [0.2, 0, 1, 0.8],
                      [0.2, 0.2, 1, 1]]

# un-normalize
__C.TRAIN.UN_NORMALIZE = False

# threshold for zoom
__C.TRAIN.Tz = 0.0

# number of proposals in training
__C.TRAIN.NUM_PROPOSALS  = 2000

#
# Testing options
#
__C.TEST = edict()

# Scales to use during testing (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TEST.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
__C.TEST.MAX_SIZE = 1000

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.NMS = 0.4

# Experimental: treat the (K+1) units in the cls_score layer as linear
# predictors (trained, eg, with one-vs-rest SVMs).
__C.TEST.SVM = False

# Test using bounding-box regression
__C.TEST.BBOX_REG = True

# whether to display results or not
__C.TEST.DISPLAY = False

# threshold for zoom
__C.TEST.Tz = 0.1

# number of proposals in training
__C.TEST.NUM_PROPOSALS  = 300

#
# Options that controls the AZ-Net search process
#

__C.SEAR = edict()

# A region is responsible for adjacent prediction if either of the case happens
#    (1) Its IOU score is above the ADJ_THRESH with an object
#    (2) It is the region with largest IOU with the object
# An object is considered embedded in a region if
#    (1) At least (EMB_AREA_THRESH) % of its area is contained in the region
#    (2) Its IoU is at most EMB_IOU_THRESH with the region

# Detection template for a region
__C.SEAR.SUBREGION = [[0,0,1,1],
                      [-0.5,0,0.5,1],[0.5,0,1.5,1],
                      [0,-0.5,1,0.5],[0,0.5,1,1.5],
                      [0,0,0.5,1],[0.5,0,1,1],
                      [0,0,1,0.5],[0,0.5,1,1],
                      [0.25,0,0.75,1], [0,0.25,1,0.75]]
__C.SEAR.NUM_SUBREG = len(__C.SEAR.SUBREGION)

# Assumed error probability in zoom indicator for training
__C.SEAR.ZOOM_ERR_PROB = 0.3
# Repetition in generating the training samples
__C.SEAR.TRAIN_REP = 8
# IOU threshold for an object to be adjacent to a region
__C.SEAR.ADJ_THRESH = 0.1
# Minimum proportion of area of an object to be considered embedded in a region
__C.SEAR.EMB_OBJ_THRESH = 0.5
# Maximum relative size of an object to be considered embedded in a region
__C.SEAR.EMB_REG_THRESH = 0.25
# Scaled prediction confidence score according to region overlapping
__C.SEAR.SCALE_ADJ_CONF = False

# threshold in confidence score
__C.SEAR.Tc = 0.005
__C.SEAR.FIXED_PROPOSAL_NUM = True

# Append boxes around proposals
__C.SEAR.APPEND_BOXES = False
__C.SEAR.APPEND_TEMP = np.transpose(np.array([[[0,0,1,1],
                                               [-0.25,0,1,1],
                                               [0,0,1.25,1],
                                               [0,-0.25,1,1],
                                               [0,0,1,1.25],
                                               [-0.125,-0.125,1.125,1.125],
                                               [0.125,0.125,0.875,0.875]]]), 
                                    axes=[0,2,1])
# The minimum lenght of side (in pixels) for a region to be considered in adajacent prediction and zoom-in
__C.SEAR.MIN_SIDE = 10

# batch size of region processing (to prevent excessive GPU memory consumption)
__C.SEAR.BATCH_SIZE = 10000

# conv layers for AZ-Net
__C.SEAR.AZ_CONV = ['conv5_3']

# conv layers for FRCNN
__C.SEAR.FRCNN_CONV = ['conv5_3']

#
# MISC
#

# The mapping from image coordinates to feature map coordinates might cause
# some boxes that are distinct in image space to become identical in feature
# coordinates. If DEDUP_BOXES > 0, then DEDUP_BOXES is used as the scale factor
# for identifying duplicate boxes.
# 1/16 is correct for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
__C.DEDUP_BOXES = 1./16.

# Pixel mean values (BGR order) as a (1, 1, 3) array
# These are the values originally used for training VGG16
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# For reproducibility
__C.RNG_SEED = 3

# A small number that's used many times
__C.EPS = 1e-14

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Place outputs under an experiments directory
__C.EXP_DIR = 'default'

def get_output_dir(imdb, net):
    """Return the directory where experimental artifacts are placed.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    path = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
    if net is None:
        return path
    else:
        return osp.join(path, net.name)

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))
        
        if k == 'PIXEL_MEANS':
            v = np.array(v)
            
        # the types must match, too
        if type(b[k]) is not type(v):
            raise ValueError(('Type mismatch ({} vs. {}) '
                              'for config key: {}').format(type(b[k]),
                                                           type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)

def cfg_set_mode(mode):
    """Set train or test mode."""
    if mode == 'Train':
        __C.SEAR.Tz = __C.TRAIN.Tz
        __C.SEAR.NUM_PROPOSALS = __C.TRAIN.NUM_PROPOSALS
    elif mode == 'Test':
        __C.SEAR.Tz = __C.TEST.Tz
        __C.SEAR.NUM_PROPOSALS = __C.TEST.NUM_PROPOSALS
