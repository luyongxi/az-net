# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""

import numpy as np
from detect.config import cfg, get_output_dir
from detect.test import im_propose
import utils.cython_bbox
import cv2
import os
import cPickle

def _flip_boxes(oldboxes, im_width):
    boxes = oldboxes.copy()
    oldx1 = boxes[:, 0].copy()
    oldx2 = boxes[:, 2].copy()
    boxes[:, 0] = im_width - oldx2 - 1
    boxes[:, 2] = im_width - oldx1 - 1
    
    return boxes

def prepare_roidb(imdb, net):
    """Enrich the imdb's roidb by adding some derived quantities that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    """
    if cfg.TRAIN.USE_FLIPPED:
        num_images = len(imdb.image_index)/2
    else:
        num_images = len(imdb.image_index)
    prop = [[] for _ in xrange(num_images)]
        
    use_loaded = False
    output_dir = get_output_dir(imdb, net['full'])
    cache_file = os.path.join(output_dir, 'proposals.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            prop = cPickle.load(fid)
        print '{} proposals loaded from {}'.format(imdb.name, cache_file)
        use_loaded = True
    
    roidb = imdb.roidb
    for i in xrange(len(imdb.image_index)):
        if i % 20 == 0:
            print 'Processing {}/{} ...'.format(i, len(imdb.image_index)) 
        
        roidb[i]['image'] = imdb.image_path_at(i)
        im_size = imdb.image_size(i)
        
        # For flipped images, load proposals and detections
        if roidb[i]['flipped']:
            im_width = im_size[1]
            index = i - num_images
            roidb[i]['ex_boxes'] = _flip_boxes(roidb[index]['ex_boxes'], im_width)\
                                        .astype(np.float32, copy=False)
            roidb[i]['gt_boxes'] = _flip_boxes(roidb[index]['gt_boxes'], im_width)\
                                        .astype(np.float32, copy=False)
            roidb[i]['gt_labels']  = roidb[index]['gt_labels']
                        
            continue
       
        # need gt_overlaps as a dense array for argmax
        gt_overlaps = roidb[i]['gt_overlaps'].toarray()    
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # find out ground truths
        gt_inds = np.where(max_overlaps == 1)[0]
        gt_rois = roidb[i]['boxes'][gt_inds, :]
        # gt class of the objects
        max_classes = gt_overlaps.argmax(axis=1)
        roidb[i]['gt_labels'] = max_classes[gt_inds]

        # use trained AZ-Net to generate region proposals
        if use_loaded:
            ex_rois = np.vstack((prop[i], gt_rois))
        else:
            im = cv2.imread(roidb[i]['image'])
            ex_rois, prop[i] = _compute_ex_rois_with_net(im, net, gt_rois)
        
        prop[i] = prop[i].astype(np.uint16, copy=False)
        roidb[i]['ex_boxes'] = ex_rois.astype(np.uint16, copy=False)
        roidb[i]['gt_boxes'] = gt_rois.astype(np.uint16, copy=False)
        
        # sanity checks
        # gt boxes => class should not be zero (must be a fg class)
        assert all(max_classes[gt_inds] != 0)
        
        assert roidb[i]['ex_boxes'].shape[0] > 0, 'no example boxes'
     
    if not use_loaded:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(prop, fid, cPickle.HIGHEST_PROTOCOL)
            print 'wrote roidb (proposals) to {}'.format(cache_file)

def add_bbox_regression_targets(roidb):
    """Add information needed to train bounding-box regressors."""
    assert len(roidb) > 0
    assert 'gt_labels' in roidb[0], 'Did you call prepare_roidb first?'

    num_images = len(roidb)
    # Infer number of classes from the number of columns in gt_overlaps
    num_classes = roidb[0]['gt_overlaps'].shape[1]
    for im_i in xrange(num_images):
        ex_rois = roidb[im_i]['ex_boxes']
        gt_rois = roidb[im_i]['gt_boxes']
        gt_labels = roidb[im_i]['gt_labels']
        roidb[im_i]['bbox_targets'], roidb[im_i]['max_overlaps'] = \
                _compute_targets(ex_rois, gt_rois, gt_labels)

    # Compute values needed for means and stds
    # var(x) = E(x^2) - E(x)^2
    class_counts = np.zeros((num_classes, 1)) + cfg.EPS
    sums = np.zeros((num_classes, 4))
    squared_sums = np.zeros((num_classes, 4))
    for im_i in xrange(num_images):
        targets = roidb[im_i]['bbox_targets']
                
        for cls in xrange(1, num_classes):
            cls_inds = np.where(targets[:, 0] == cls)[0]
            if cls_inds.size > 0:
                class_counts[cls] += cls_inds.size
                sums[cls, :] += targets[cls_inds, 1:].sum(axis=0)
                squared_sums[cls, :] += (targets[cls_inds, 1:] ** 2).sum(axis=0)

    means = sums / class_counts
    stds = np.sqrt(squared_sums / class_counts - means ** 2)

    # Normalize targets
    for im_i in xrange(num_images):
        targets = roidb[im_i]['bbox_targets']
        for cls in xrange(1, num_classes):
            cls_inds = np.where(targets[:, 0] == cls)[0]
            roidb[im_i]['bbox_targets'][cls_inds, 1:] -= means[cls, :]
            roidb[im_i]['bbox_targets'][cls_inds, 1:] /= stds[cls, :]

    # These values will be needed for making predictions
    # (the predicts will need to be unnormalized and uncentered)
    return means.ravel(), stds.ravel()

def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""
    # Ensure ROIs are floats
    ex_rois = ex_rois.astype(np.float, copy=False)
    gt_rois = gt_rois.astype(np.float, copy=False)

    # Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = utils.cython_bbox.bbox_overlaps(ex_rois,
                                                     gt_rois)

    # Indices of examples for which we try to make predictions
    if gt_rois.shape[0] == 0:
        max_overlaps = cfg.TRAIN.BG_THRESH_LO * np.ones((ex_rois.shape[0],), dtype=np.float32)
        targets = np.zeros((ex_rois.shape[0], 5), dtype=np.float32)
        return targets, max_overlaps
    else:
        max_overlaps = ex_gt_overlaps.max(axis=1)
    pos_inds = np.where(max_overlaps >= cfg.TRAIN.BBOX_THRESH)[0]

    # Find which gt ROI each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    gt_assignment = ex_gt_overlaps.argmax(axis=1)
    # target rois
    tar_rois = gt_rois[gt_assignment[pos_inds], :]
    # positive examples
    pos_rois = ex_rois[pos_inds, :]

    pos_widths = pos_rois[:, 2] - pos_rois[:, 0] + cfg.EPS
    pos_heights = pos_rois[:, 3] - pos_rois[:, 1] + cfg.EPS
    pos_ctr_x = pos_rois[:, 0] + 0.5 * pos_widths
    pos_ctr_y = pos_rois[:, 1] + 0.5 * pos_heights

    tar_widths = tar_rois[:, 2] - tar_rois[:, 0] + cfg.EPS
    tar_heights = tar_rois[:, 3] - tar_rois[:, 1] + cfg.EPS
    tar_ctr_x = tar_rois[:, 0] + 0.5 * tar_widths
    tar_ctr_y = tar_rois[:, 1] + 0.5 * tar_heights
    
    pos_widths = np.maximum(1, pos_widths)
    pos_heights = np.maximum(1, pos_heights)
    tar_widths = np.maximum(1, tar_widths)
    tar_heights = np.maximum(1, tar_heights)

    targets_dx = (tar_ctr_x - pos_ctr_x) / pos_widths
    targets_dy = (tar_ctr_y - pos_ctr_y) / pos_heights
    targets_dw = np.log(tar_widths / pos_widths)
    targets_dh = np.log(tar_heights / pos_heights)

    targets = np.zeros((ex_rois.shape[0], 5), dtype=np.float32)
    targets[pos_inds, 0] = labels[gt_assignment[pos_inds]]
    targets[pos_inds, 1] = targets_dx
    targets[pos_inds, 2] = targets_dy
    targets[pos_inds, 3] = targets_dw
    targets[pos_inds, 4] = targets_dh
    return targets, max_overlaps

def _compute_ex_rois_with_net(im, net, gt_rois):
    """ Generate RoIs by using zoom in and adjacency predictions
    """    
    # regions generated from SC-Net
    regions = im_propose(net, im, 
                         num_proposals = cfg.SEAR.NUM_PROPOSALS)
    return np.vstack((regions, gt_rois)), regions
