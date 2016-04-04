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

"""Compute minibatch blobs for training AZ-Net."""

import numpy as np
import numpy.random as npr
import cv2
from detect.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob

def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = np.round(cfg.TRAIN.AZ_POS_FRACTION * rois_per_image)

    # Get the input image blob, formatted for caffe
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    # Now, build the region of interest and label blobs
    rois_blob = np.zeros((0, 5), dtype=np.float32)
    adj_labels_blob = np.zeros((0, num_classes), dtype=np.float32)
    adj_targets_blob = np.zeros((0, 4 * num_classes), dtype=np.float32)
    adj_loss_blob = np.zeros(adj_targets_blob.shape, dtype=np.float32)
    zoom_labels_blob = np.zeros((0), dtype=np.float32)
    # all_overlaps = []
    for im_i in xrange(num_images):
        adj_labels, zoom_labels, im_rois, adj_targets, adj_loss \
            = _sample_rois(roidb[im_i], fg_rois_per_image, rois_per_image)

        # Add to RoIs blob
        rois = _project_im_rois(im_rois, im_scales[im_i])
        batch_ind = im_i * np.ones((rois.shape[0], 1))
        rois_blob_this_image = np.hstack((batch_ind, rois))
        rois_blob = np.vstack((rois_blob, rois_blob_this_image))

        # Add to labels_pred, bbox targets, bbox loss blobs, and labels_zoom box
        adj_labels_blob = np.vstack((adj_labels_blob, adj_labels))
        adj_targets_blob = np.vstack((adj_targets_blob, adj_targets))
        adj_loss_blob = np.vstack((adj_loss_blob, adj_loss))
        zoom_labels_blob = np.hstack((zoom_labels_blob, zoom_labels))

    # For debug visualizations
    # _vis_minibatch(im_blob, rois_blob, labels_blob, all_overlaps)

    blobs = {'data': im_blob,
             'rois': rois_blob,
             'adj_labels': adj_labels_blob,
             'adj_targets': adj_targets_blob,
             'adj_loss_weights': adj_loss_blob, 
             'zoom_labels': zoom_labels_blob}

    return blobs

def _sample_rois(roidb, fg_rois_per_image, rois_per_image):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """   
    zoom_labels = roidb['zoom_gt']
    rois = roidb['ex_boxes'].astype(np.float32, copy=False)
    
    adj_labels = np.zeros((rois.shape[0], cfg.SEAR.NUM_SUBREG))
    adj_matching = roidb['bbox_targets'][:, 4:6].astype(np.uint32, copy=False)
    IOU_target = roidb['bbox_targets'][:, -1]
    for cls in xrange(cfg.SEAR.NUM_SUBREG):
        cls_inds = np.where(adj_matching[:, 1] == cls)[0]
        if not cfg.SEAR.SCALE_ADJ_CONF:
            adj_labels[adj_matching[cls_inds, 0], cls] = 1
        else:
            adj_labels[adj_matching[cls_inds, 0], cls] = \
               IOU_target[cls_inds]
                
    # Find foreground RoIs
    fg_inds = np.where((adj_labels.any(axis=1) == 1) |
                       (zoom_labels == 1))[0]

    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image,
                             replace=False)

    # Find background RoIs
    bg_inds = np.where((adj_labels.any(axis=1) == 0) |
                       (zoom_labels == 0))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                        bg_inds.size)
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image,
                             replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sample values from various arrays
    adj_labels = adj_labels[keep_inds]
    zoom_labels = zoom_labels[keep_inds]
    
    rois = rois[keep_inds]

    adj_targets, adj_loss_weights = \
        _get_adjacent_targets(roidb['bbox_targets'], 
                              keep_inds, 
                              roidb['ex_boxes'].shape[0],
                              cfg.SEAR.NUM_SUBREG)

    return adj_labels, zoom_labels, rois, adj_targets, adj_loss_weights

def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in xrange(num_images):
        im = cv2.imread(roidb[i]['image'])
        assert (im is not None), 'image not found: {0}'.format(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales

def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois

def _get_adjacent_targets(compact_targets, keep_inds, num_regions, num_classes):
    """Get adjacent prediction labels
    """
    bbox_targets = np.zeros((num_regions, 4 * num_classes), dtype=np.float32)
    bbox_loss_weights = np.zeros((num_regions, 4 * num_classes), dtype=np.float32)
        
    for cls in xrange(num_classes):
        start = 4 * cls
        end = start + 4        
        cls_inds = np.where(compact_targets[:, -2] == cls)[0]
        reg_inds = compact_targets[cls_inds, -3]        
        for i in xrange(len(reg_inds)):
            ind = reg_inds[i]            
            bbox_targets[ind, start:end] = compact_targets[cls_inds[i], 0:4]
            bbox_loss_weights[ind, start:end] = [1., 1., 1., 1.]
            
    return bbox_targets[keep_inds], bbox_loss_weights[keep_inds]

def _vis_minibatch(im_blob, rois_blob, labels_blob, overlaps):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
    for i in xrange(rois_blob.shape[0]):
        rois = rois_blob[i, :]
        im_ind = rois[0]
        roi = rois[1:]
        im = im_blob[im_ind, :, :, :].transpose((1, 2, 0)).copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        cls = labels_blob[i]
        plt.imshow(im)
        print 'class: ', cls, ' overlap: ', overlaps[i]
        plt.gca().add_patch(
            plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                          roi[3] - roi[1], fill=False,
                          edgecolor='r', linewidth=3)
            )
        plt.show()
