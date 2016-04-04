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

"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""

import numpy as np
import numpy.random as npr
from detect.config import cfg
from detect.test import divide_region
from utils.cython_bbox import bbox_overlaps, bbox_zoom_labels
import os, sys
import cPickle
import cv2

def prepare_roidb(imdb):
    """Enrich the imdb's roidb by adding some derived quantities that
    are useful for training. 
    """
    cache_file = os.path.join(imdb.cache_path, 
                              imdb.name + '_trainable_roidb.pkl')
    use_cache = False
    if cfg.TRAIN.USE_CACHE and os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            caches = cPickle.load(fid)
            zoom_gt = caches['zoom_gt']
	    ex_boxes = caches['ex_boxes']
	    gt_boxes = caches['gt_boxes']
            use_cache = True
        print '{} trainable caches loaded from {}'.format(imdb.name, cache_file)
    
    roidb = imdb.roidb
    for i in xrange(len(imdb.image_index)):
        if i % 10000 == 0:
            print 'Processing {}/{} ...'.format(i, len(imdb.image_index))        
        
        roidb[i]['image'] = imdb.image_path_at(i)
       
	if use_cache:
	    roidb[i]['zoom_gt'] = zoom_gt[i]
	    roidb[i]['ex_boxes'] = ex_boxes[i]
	    roidb[i]['gt_boxes'] = gt_boxes[i]
	else:
	    # need gt_overlaps as a dense array for argmax
            gt_overlaps = roidb[i]['gt_overlaps'].toarray()        
            # max overlap with gt over classes (columns)
            max_overlaps = gt_overlaps.max(axis=1)
            # find out ground truths
            gt_inds = np.where(max_overlaps == 1)[0]
            gt_boxes = roidb[i]['boxes'][gt_inds, :]
        
            # generate training rois, based on
            #    (1) image size
            #    (2) ground truth boxes
            ex_boxes, zoom_gt = _compute_ex_rois(imdb.image_size(i), gt_boxes)
        
            roidb[i]['zoom_gt'] = zoom_gt.astype(np.bool, copy=False)
            roidb[i]['ex_boxes'] = ex_boxes.astype(np.float32, copy=False)
            roidb[i]['gt_boxes'] = gt_boxes.astype(np.float32, copy=False)
        
        assert np.all(roidb[i]['ex_boxes'][:, 0] <= roidb[i]['ex_boxes'][:, 2] ), 'error in ex_width id={0}'.format(i)
        assert np.all(roidb[i]['ex_boxes'][:, 1] <= roidb[i]['ex_boxes'][:, 3] ), 'error in ex_height id={0}'.format(i)
        assert np.all(roidb[i]['gt_boxes'][:, 0] <= roidb[i]['gt_boxes'][:, 2] ), 'error in gt_width id={0}'.format(i)
        assert np.all(roidb[i]['gt_boxes'][:, 1] <= roidb[i]['gt_boxes'][:, 3] ), 'error in gt_height id={0}'.format(i)
    
    if cfg.TRAIN.USE_CACHE and (not os.path.exists(cache_file)):
        with open(cache_file, 'wb') as fid:
	    caches = {'zoom_gt': zoom_gt, 'ex_boxes': ex_boxes, 'gt_boxes': gt_boxes}
            cPickle.dump(caches, fid, cPickle.HIGHEST_PROTOCOL)
            print 'wrote trainable caches to {}'.format(cache_file)

def add_adjacent_prediction_targets(imdb):
    """Add information needed to train adjacency predictors."""
    cache_file = os.path.join(imdb.cache_path, 
                              imdb.name + '_targets_roidb.pkl')
    use_cache = False
    if cfg.TRAIN.USE_CACHE and os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            caches = cPickle.load(fid)
            bbox_targets = caches['bbox_targets']
	    means = caches['means']
            stds = caches['stds']
	    use_cache = True
        print '{} targets cache loaded from {}'.format(imdb.name, cache_file)
    
    roidb = imdb.roidb
    assert len(roidb) > 0
    assert 'zoom_gt' in roidb[0], 'Did you call prepare_roidb first?'

    num_images = len(roidb)
    num_classes = cfg.SEAR.NUM_SUBREG
    
    for im_i in xrange(num_images):
        if im_i % 10000 == 0:
            print 'Processing {}/{} ...'.format(im_i, num_images)
	if use_cache:
            roidb[im_i]['bbox_targets'] = bbox_targets[im_i]
	else:
	    gt_rois = roidb[im_i]['gt_boxes']
            ex_rois = roidb[im_i]['ex_boxes']
            roidb[im_i]['bbox_targets'] = _compute_targets(gt_rois, ex_rois)
    
    # Compute values needed for means and stds
    # var(x) = E(x^2) - E(x)^2
    if not use_cache:
    	class_counts = np.zeros((num_classes, 1)) + cfg.EPS
    	sums = np.zeros((num_classes, 4))
    	squared_sums = np.zeros((num_classes, 4))
    	for im_i in xrange(num_images):
            targets = roidb[im_i]['bbox_targets']
       	    for cls in xrange(num_classes):
                cls_inds = np.where(targets[:,-2] == cls)[0]
            	if len(cls_inds) > 0:
                    class_counts[cls] += len(cls_inds)            
                    sums[cls, :] += targets[cls_inds, 0:4].sum(axis=0)
                    squared_sums[cls, :] += (targets[cls_inds, 0:4] ** 2).sum(axis=0)

        means = sums / class_counts
        stds = np.sqrt(squared_sums / class_counts - means ** 2)

	# Normalize targets
        for im_i in xrange(num_images):
            targets = roidb[im_i]['bbox_targets']
            for cls in xrange(num_classes):
                cls_inds = np.where(targets[:,-2] == cls)[0]
                roidb[im_i]['bbox_targets'][cls_inds, 0:4] -= means[cls, :]
                roidb[im_i]['bbox_targets'][cls_inds, 0:4] /= stds[cls, :]
            
    if cfg.TRAIN.USE_CACHE and (not os.path.exists(cache_file)):
        with open(cache_file, 'wb') as fid:
            caches = {'bbox_targets': bbox_targets, 'means': means, 'stds': stds}
            cPickle.dump(caches, fid, cPickle.HIGHEST_PROTOCOL)
            print 'wrote targets cache to {}'.format(cache_file)

    # These values will be needed for making predictions
    # (the predicts will need to be unnormalized and uncentered)
    return means.ravel(), stds.ravel()

def _compute_targets(gt_rois, ex_rois):
    """Compute bounding-box regression targets for an image.
        gt_rois: ground truth rois
        ex_rois: example rois
    """
    K = ex_rois.shape[0]
    N = gt_rois.shape[0]
    # Ensure ROIs are floats
    gt_rois = gt_rois.astype(np.float, copy=False)
    ex_rois = ex_rois.astype(np.float, copy=False)

    # bbox targets: (x1,y1,x2,y2,ex_rois_ind,subreg_ind)  
    targets = np.zeros((0, 7), dtype=np.float32)
    
    if K == 0 or N == 0:
        return targets
    
    # For each region, find out objects that are adjacent
    # Match objects to sub-regions with maximum overlaps. 
    # Objects with large overlaps with any sub-regions are given priority.
    overlaps = bbox_overlaps(ex_rois, gt_rois)
    max_overlaps = overlaps.max(axis=1)

    for k in xrange(K):
        
        if max_overlaps[k] < cfg.SEAR.ADJ_THRESH:
            continue
        
        re = ex_rois[k, :]
        L = np.array([[re[2]-re[0], re[3]-re[1], re[2]-re[0], re[3]-re[1]]]) 
        delta = np.array([[re[0], re[1], re[0], re[1]]])
        # sub-regions`
        s_re = (L * cfg.SEAR.SUBREGION) + delta
        s_re = s_re.astype(np.float, copy=False)
        # compute the overlaps between sub-regions and each objects
        sre_gt_overlaps = bbox_overlaps(s_re, gt_rois)
        # find out the objects that are actually adjacent
        adj_th = (sre_gt_overlaps[0] >= cfg.SEAR.ADJ_THRESH)
        match_inds = np.where(adj_th)[0]
        sre_gt_overlaps[:, ~adj_th] = -1
#        adj_th = (sre_gt_overlaps >= cfg.SEAR.ADJ_THRESH)
#        match_inds = np.where(np.any(adj_th, axis=0))[0]
        if match_inds.shape[0]>0:    # there is object to match
            for _ in xrange(min(cfg.SEAR.NUM_SUBREG, match_inds.shape[0])):            
                reg_idx, gt_idx = np.unravel_index(sre_gt_overlaps.argmax(), 
                                                   sre_gt_overlaps.shape)
                
                # no more valid match
#                if sre_gt_overlaps[reg_idx, gt_idx] < cfg.SEAR.ADJ_THRESH:
#                    break
                t_ki = _compute_bbox_deltas(ex_rois[[k], :],
                                            gt_rois[[gt_idx], :])                
                new_target = np.hstack((t_ki, np.array([[k, reg_idx, overlaps[k, gt_idx]]])))
                targets = np.vstack((targets, new_target))                

                sre_gt_overlaps[reg_idx, :] = -1
                sre_gt_overlaps[:, gt_idx] = -1

    return targets

def _compute_bbox_deltas(ex, gt):
    ex_widths = np.maximum(ex[:, [2]] - ex[:, [0]], 1) + cfg.EPS
    ex_heights = np.maximum(ex[:, [3]] - ex[:, [1]], 1) + cfg.EPS
    ex_ctr_x = ex[:, [0]] + 0.5 * ex_widths
    ex_ctr_y = ex[:, [1]] + 0.5 * ex_heights

    gt_widths = np.maximum(gt[:, [2]] - gt[:, [0]],1) + cfg.EPS
    gt_heights = np.maximum(gt[:, [3]] - gt[:, [1]], 1) + cfg.EPS
    gt_ctr_x = gt[:, [0]] + 0.5 * gt_widths
    gt_ctr_y = gt[:, [1]] + 0.5 * gt_heights
    
    ex_widths = np.maximum(1, ex_widths)
    ex_heights = np.maximum(1, ex_heights)
    gt_widths = np.maximum(1, gt_widths)
    gt_heights = np.maximum(1, gt_heights)

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)
    
    return np.hstack((targets_dx, targets_dy, targets_dw, targets_dh))
    

def _compute_ex_rois(size, gt_rois):
    """ Generate RoIs by zoom in to ideal grid (with random disturbances)
    """
    # the regions that are selected
    Bsel = np.zeros((0,4))
    # the labels for zoom
    zoom_labels = np.zeros((0,))
    # the current set of regions
    w = size[1] - 1.0
    h = size[0] - 1.0
    lengths = np.array([[w,h,w,h]])
    for _ in xrange(cfg.SEAR.TRAIN_REP):
        # the current set of regions
        B = lengths * cfg.TRAIN.ADDREGIONS
        # the set for zoom in
        Z = B
        # number of layers for the search
        height = size[0]
        width = size[1]
        side = np.minimum(height, width)
        K = int(np.log2(side/cfg.SEAR.MIN_SIDE) + 1.0)
        for _ in xrange(K):
            # compute zoom labels
            zScores = _compute_zoom_labels(B, gt_rois)
            # selected regions
            Bsel = np.vstack((Bsel, B))
            # zoom labels
            zoom_labels = np.hstack((zoom_labels, zScores))
            # error vector
            err = (npr.random(size=B.shape[0]) <= 
                   cfg.SEAR.ZOOM_ERR_PROB)
            # decide where to zoom
            indZ = np.where(np.logical_xor(zScores, err))[0]
            # Z is updated to regions for zoom in
            Z = B[indZ, :]
            if Z.shape[0] == 0:
                break
            # B is updated to be regions that are expanded from it
            B = divide_region(Z)

    # add positive examples for sub-regions
    for n in xrange(gt_rois.shape[0]):
        ri = gt_rois[n, :]
        # lengths of the RoI
        li = np.array([[ri[2]-ri[0]+1.0,ri[3]-ri[1]+1.0]])
        # scale lengths of the sub-region templates
        rt = np.array(cfg.SEAR.SUBREGION)
        lt = np.hstack((rt[:,[2]]-rt[:,[0]], rt[:,[3]]-rt[:,[1]]))
        # target super region lengths
        ls = li/lt
        # super region top-left corners
        ts = np.hstack((ri[0]-ls[:, [0]]*rt[:,[0]], ri[1]-ls[:, [1]]*rt[:,[1]]))
        # super region coordinates
        rs = np.hstack((ts, ts[:,[0]]+ls[:,[0]]-1, 
                        ts[:,[1]]+ls[:,[1]]-1))
                      
        Bsel = np.vstack((Bsel, rs))
        zoom_labels = \
            np.hstack((zoom_labels, 
                       _compute_zoom_labels(rs, gt_rois)))
    
    # clip boxes to ensure they are within the boundary    
    Bsel = _clip_boxes(Bsel, size)
    
    heights = Bsel[:,3] - Bsel[:,1] + 1
    widths = Bsel[:,2] - Bsel[:,0] + 1
    sides = np.minimum(heights, widths)
    keep_inds = np.where(sides >= cfg.SEAR.MIN_SIDE)[0]

    return Bsel[keep_inds, :], zoom_labels[keep_inds]

def _clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes

def _compute_zoom_labels(rois, gt_rois):
    """ Compute the zoom labels for each region given the gt RoIs
    """    
    rois = rois.astype(np.float, copy=False)
    gt_rois = gt_rois.astype(np.float, copy=False)
    
#    prop = bbox_prop(regions, gt_rois)
#    overlaps = bbox_overlaps(regions, gt_rois)
    
#    emb_labels = np.any((prop >= cfg.SEAR.EMB_AREA_THRESH) & 
#                         (overlaps <= cfg.SEAR.EMB_IOU_THRESH), 
#                         axis=1)
#    group_labels = (np.sum(prop >= cfg.SEAR.EMB_AREA_THRESH, 
#                           axis=1) > 1)
    
#    zoom_labels = (emb_labels | group_labels)
#    overlaps = bbox_asm_overlaps(regions, gt_rois)
    
#    bbox_area = (regions[:,3]-regions[:,1]+1.0) * (regions[:,2]-regions[:,0]+1.0)
#    gt_area = (gt_rois[:,3]-gt_rois[:,1]+1.0) * (gt_rois[:,2]-gt_rois[:,0]+1.0)
#    area_ratio = gt_area[np.newaxis, :]/bbox_area[:, np.newaxis]
    
    max_area_ratio = np.float(cfg.SEAR.EMB_REG_THRESH)
    overlaps, area_ratio = bbox_zoom_labels(rois, gt_rois, max_area_ratio) 
    zoom_labels = np.any((area_ratio <= cfg.SEAR.EMB_REG_THRESH) & 
                         (overlaps >= cfg.SEAR.EMB_OBJ_THRESH),
                         axis=1)
    
    return zoom_labels
