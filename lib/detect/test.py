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

"""Test using SC-Net on an imdb (image database)."""

from detect.config import cfg, get_output_dir
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from utils.cython_nms import nms
from utils.cython_bbox import bbox_overlaps
import utils.cython_div as div
import cPickle
import heapq
from utils.blob import im_list_to_blob
import os

def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
        im (ndarray): a color image in BGR order
    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)
    
    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.
    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob
    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.
    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob
    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors

def _bbox_pred(boxes, box_deltas):
    """Transform the set of class-agnostic boxes into class-specific boxes
    by applying the predicted offsets (box_deltas)
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + cfg.EPS
    heights = boxes[:, 3] - boxes[:, 1] + cfg.EPS
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = box_deltas[:, 0::4]
    dy = box_deltas[:, 1::4]
    dw = box_deltas[:, 2::4]
    dh = box_deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(box_deltas.shape)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

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

def divide_region(regions):
    """ Divide the regions into non-overlapping sub-regions
        The algorithm first finds the shorter side of a region,
        it then divides the image along that axis into 2 parts.
        Then it finds the closest division along the longer axis
        so that the generated regions are close to squares
    """        
    regions.astype(np.float, copy=False)
    return div.divide_region(regions, np.float(cfg.SEAR.MIN_SIDE))
        
def _sift_small(regions):
    """ Sift away regions with less-than-minimum height
    """        
    heights = regions[:,3]-regions[:,1]+1.0
    keep_inds = np.where(heights >= cfg.SEAR.MIN_HEIGHT)[0]
    
    return regions[keep_inds, :]

def _unwrap_adj_pred(boxes, scores):
    """ Unwrap the adjacent predictions to 
        vector form, sifting out boxes that are too small
    """
    scores = scores.ravel()
    x1 = boxes[:, 0::4].ravel()
    y1 = boxes[:, 1::4].ravel()
    x2 = boxes[:, 2::4].ravel()
    y2 = boxes[:, 3::4].ravel() 
    boxes = np.vstack((x1,y1,x2,y2)).transpose()
    
    heights = boxes[:,3] - boxes[:,1] + 1
    widths = boxes[:, 2] - boxes[:,0] + 1
    sides = np.minimum(heights, widths)
    keep_inds = np.where(sides >= cfg.SEAR.MIN_SIDE)[0]
    
    return boxes[keep_inds, :], scores[keep_inds]
    
def _az_forward(net, im, all_boxes, conv = None):
    """ Forward the AZ-network
        To prevent excessive GPU memory consumption, 
        ROI pooling is performed in batches if necessary
    """
    conv_name = cfg.SEAR.LAST_CONV
    
    batchSize = cfg.SEAR.BATCH_SIZE
    num_batches = int(np.ceil(all_boxes.shape[0] / float(batchSize)))
    
    zScores = np.zeros((0,))
    aBBox = np.zeros((0,4))
    cScores = np.zeros((0,))
    
    for bid in xrange(num_batches):
        start = batchSize * bid
        end = min(all_boxes.shape[0], batchSize * (bid+1))
        boxes = all_boxes[start:end, :]
    
        boxes = boxes[:, 0:4]
        blobs, unused_im_scale_factors = _get_blobs(im, boxes)
        
        # remove duplicates in feature space
        inv_index = None
        if cfg.DEDUP_BOXES > 0:
            v = np.array([1, 1e3, 1e6, 1e9, 1e12])
            hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
            _, index, inv_index = np.unique(hashes, return_index=True, 
                                return_inverse = True)
            blobs['rois'] = blobs['rois'][index, :]
            boxes = boxes[index, :]
    
        # forward network (actual action depends on whether the convolutional layers are provided)    
        if conv == None or 'fc' not in net.keys():
            net['full'].blobs['data'].reshape(*(blobs['data'].shape))
            net['full'].blobs['rois'].reshape(*(blobs['rois'].shape))
            blobs_out = net['full'].forward(data=blobs['data'].astype(np.float32, copy=False),
                                            rois=blobs['rois'].astype(np.float32, copy=False),
                                            blobs = [conv_name])
            conv = blobs_out[conv_name]
        else:
            net['fc'].blobs['conv'].reshape(*(conv.shape))
            net['fc'].blobs['rois'].reshape(*(blobs['rois'].shape))
            blobs_out = net['fc'].forward(conv=conv.astype(np.float32, copy=False),
                                            rois=blobs['rois'].astype(np.float32, copy=False))
            
        z_tb = blobs_out['zoom_prob'] 
         
        # adjacent predictions  
        pred_scores = blobs_out['adj_prob']
        box_deltas = blobs_out['adj_bbox']
        pred_boxes = _bbox_pred(boxes, box_deltas)
        pred_boxes = _clip_boxes(pred_boxes, im.shape)

        if cfg.DEDUP_BOXES > 0:
            pred_scores = pred_scores[inv_index, :]
            pred_boxes = pred_boxes[inv_index, :]
            z_tb = z_tb[inv_index].ravel()
    
        a_tb, c_tb = _unwrap_adj_pred(pred_boxes, pred_scores)
        
        zScores = np.hstack((zScores, z_tb))
        aBBox = np.vstack((aBBox, a_tb))
        cScores = np.hstack((cScores, c_tb))
    
    return zScores, aBBox, cScores, conv

def _frcnn_forward(net, im, all_boxes, num_classes, conv = None):
    """ Forward the Fast R-CNN network
        To prevent excessive GPU memory consumption, 
        ROI pooling is performed in batches if necessary
    """
    conv_name = cfg.SEAR.LAST_CONV
    
    batchSize = cfg.SEAR.BATCH_SIZE
    num_batches = int(np.ceil(all_boxes.shape[0] / float(batchSize)))
    
    all_pred_boxes = np.zeros((0, 4*num_classes))
    all_scores = np.zeros((0, num_classes))
    
    for bid in xrange(num_batches):
        start = batchSize * bid
        end = min(all_boxes.shape[0], batchSize * (bid+1))
        boxes = all_boxes[start:end, :]
    
        boxes = boxes[:, 0:4]
        blobs, unused_im_scale_factors = _get_blobs(im, boxes)
        
        # remove duplicates in feature space
        inv_index = None
        if cfg.DEDUP_BOXES > 0:
            v = np.array([1, 1e3, 1e6, 1e9, 1e12])
            hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
            _, index, inv_index = np.unique(hashes, return_index=True, 
                                return_inverse = True)
            blobs['rois'] = blobs['rois'][index, :]
            boxes = boxes[index, :]
    
        # forward network (actual action depends on whether the convolutional layers are provided)    
        if conv == None or 'fc' not in net.keys():
            net['full'].blobs['data'].reshape(*(blobs['data'].shape))
            net['full'].blobs['rois'].reshape(*(blobs['rois'].shape))
            blobs_out = net['full'].forward(data=blobs['data'].astype(np.float32, copy=False),
                                            rois=blobs['rois'].astype(np.float32, copy=False),
                                            blobs = [conv_name])
            conv = blobs_out[conv_name]
        else:
            net['fc'].blobs['conv'].reshape(*(conv.shape))
            net['fc'].blobs['rois'].reshape(*(blobs['rois'].shape))
            blobs_out = net['fc'].forward(conv=conv.astype(np.float32, copy=False),
                                            rois=blobs['rois'].astype(np.float32, copy=False))
            
        # adjacent predictions  
        pred_scores = blobs_out['cls_prob']
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = _bbox_pred(boxes, box_deltas)
        pred_boxes = _clip_boxes(pred_boxes, im.shape)

        if cfg.DEDUP_BOXES > 0:
            pred_scores = pred_scores[inv_index, :]
            pred_boxes = pred_boxes[inv_index, :]
        
        all_scores = np.vstack((all_scores, pred_scores))
        all_pred_boxes = np.vstack((all_pred_boxes, pred_boxes))
    
    return all_scores, all_pred_boxes, conv

def _append_boxes(boxes):
    """ Append boxes around the input boxes, according to the templates
    """
    num_boxes = boxes.shape[0]
    num_subregs = cfg.SEAR.APPEND_TEMP.shape[2]
    
    widths = boxes[:,[2]] - boxes[:, [0]]
    heights= boxes[:, [3]] - boxes[:, [1]]
    
    L = np.hstack((widths, heights, widths, heights))
    delta = np.hstack((boxes[:,[0]], boxes[:,[1]], boxes[:,[0]], boxes[:,[1]]))
    
    L = L[:,:,np.newaxis]
    delta = delta[:,:,np.newaxis]
    
    # sub-regions
    subs = (L * cfg.SEAR.APPEND_TEMP) + delta
    subs = np.transpose(subs, [2,0,1])
    subs = subs.reshape((num_boxes * num_subregs, 4))
    
    # remove duplicates
    subs = subs.astype(np.float, copy=False)
    subs = div._sift_dup(subs, 1/cfg.DEDUP_BOXES)
    
    return subs

def im_propose(net, im, return_conv = False, num_proposals = None):
    """Generate object proposals using AZ-Net
    Arguments:
        net (caffe.Net): AZ-Net model
        im (ndarray): color image to test (in BGR order)
    Returns:
        Y (ndarray): R X 5 array of proposal
    """
    # the current set of regions
    B = np.array([[0, 0, im.shape[1]-1.0, im.shape[0]-1.0]])
    # the set for zoom in
    Z = np.vstack((B, divide_region(B)))
    # the set of region proposals from adjacent predictions
    Y = np.zeros((0, 4))
    # confidence scores of adjacent predictions
    aScores = np.zeros((0,))
    # number of evaluations
    num_eval = 0
    # number of layers for the search
    height = im.shape[0]
    width = im.shape[1]
    side = np.minimum(height, width)
    K = int(np.log2(side/cfg.SEAR.MIN_SIDE) + 1.0)
    # threshold at zoom indicator
    Tz = cfg.SEAR.Tz
    # cached convolutional layer (the last layer)
    conv = None
    for k in xrange(1, K):
        # Get zoom indicator and adjacent predictions (w/ confidence)
        # Scores and aBBox already sifted and vectorized by sc_net function
#        print B.shape[0]
        zoom, boxes, c, conv = _az_forward(net, im, B, conv)
        num_eval = num_eval + B.shape[0]
        # predictions with high confidence scores are included
        Y = np.vstack((Y, boxes))
        aScores = np.hstack((aScores, c))
        # Z is updated to regions for zoom in
        if k==1:    # heuristic: the root region is always divided
            zoom[0] = 1.0
                    
        indZ = np.where(zoom >= Tz)[0]
        Z = B[indZ, :]
        if Z.shape[0] == 0:
            break
        # B is updated to be regions that are expanded from it
        B = divide_region(Z)
        
    if (cfg.SEAR.FIXED_PROPOSAL_NUM == False) and (num_proposals is None):
        indA = np.where(aScores >= cfg.SEAR.Tc)[0]
        Y = Y[indA, :]
    else:
        if num_proposals is None:
            num_proposals = cfg.SEAR.NUM_PROPOSALS
        indA = np.argsort(-aScores)
        max_num = np.minimum(num_proposals, Y.shape[0])
        Y = Y[indA[:max_num], :]

    # append boxes
    if cfg.SEAR.APPEND_BOXES:
        Y = _append_boxes(Y)
        Y = _clip_boxes(Y, im.shape)

    print '{0} proposals, evaluate {1} regions, reaches depth {2}.'\
        .format(Y.shape[0], num_eval,k)

    if return_conv:
        return Y, conv
    else:
        return Y

def im_detect(net, im, boxes, num_classes):
    """Detect object classes in an image given object proposals using Fast R-CNN
    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals
    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        pred_boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """ 
    scores, pred_boxes, _ = \
        _frcnn_forward(net, im, boxes, num_classes)

    return scores, pred_boxes

def im_detect_shared(az_net, frcnn_net, im, num_classes):
    """Detection using AZ-Net and Fast R-CNN w/ shared convolutional layers
    Arguments:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        pred_boxes (ndarray): R x (4*K) array of predicted bounding boxes
    Returns:
        Y (ndarray): R X 5 array of proposal
    """
    boxes, conv = im_propose(az_net, im, return_conv = True)
    scores, pred_boxes, _ = \
        _frcnn_forward(frcnn_net, im, boxes, num_classes, conv)

    return scores, pred_boxes

def vis_detections(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
#    for i in xrange(np.minimum(10, dets.shape[0])):
    for i in xrange(dets.shape[0]):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()

def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            keep = nms(dets, thresh)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes

def test_proposals(net, imdb):
    """Generate proposals using AZ-Net on an image database."""
    num_images = len(imdb.image_index)
    # all proposals are collected into:
    #    prpo_boxes[cls] = N x 5 array of proposals in
    #    (x1, y1, x2, y2, score)
    prop_boxes = [[] for _ in xrange(num_images)]

    output_dir = get_output_dir(imdb, net['full'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # timers
    _t = {'im_prop' : Timer()}
    
    # initialize counters
    num_boxes = 0.0
    num_gt = 0.0
    num_det = 0.0
    
#    gt_roidb = imdb.gt_roidb()

    for i in xrange(num_images):
        im = cv2.imread(imdb.image_path_at(i))            
        _t['im_prop'].tic()
        prop_boxes[i] = \
            im_propose(net, im)
        _t['im_prop'].toc()
    
#        gt_boxes = gt_roidb[i]['boxes']
        
#        if not comp_mode:
#            if prop_boxes[i].shape[0] > 0:
#                overlaps = bbox_overlaps(prop_boxes[i].astype(np.float), 
#                                         gt_boxes.astype(np.float))
#                det_inds = np.where(np.max(overlaps, axis=0) >= 0.5)[0] 
#                num_det += len(det_inds)
            
#            num_gt += gt_boxes.shape[0]
#            num_boxes += prop_boxes[i].shape[0]

        print 'im_prop: {:d}/{:d} {:.3f}s' \
              .format(i + 1, num_images, _t['im_prop'].average_time)

#    recall = num_det / num_gt
    recall = 0
    prop = {'boxes': prop_boxes, 'time': _t['im_prop'].average_time, 'recall': recall}
    prop_file = os.path.join(output_dir, 'proposals.pkl')
    with open(prop_file, 'wb') as f:
        cPickle.dump(prop, f, cPickle.HIGHEST_PROTOCOL)
    
    print 'The recall is {:.3f}'.format(recall)
    print 'On average, {0} boxes per image are generated'.format(num_boxes/num_images)
    print 'The average proposal generation time is {:.3f}s'.format(_t['im_prop'].average_time)

def test_net(net, prop_file, imdb):
    """Test a Fast R-CNN network on an image database."""
    # load saved object proposals
    with open(prop_file, 'rb') as f:
        prop = cPickle.load(f)
        prop_boxes = prop['boxes']
    num_images = len(imdb.image_index)
    # heuristic: keep an average of 40 detections per class per images prior
    # to NMS
#    max_per_set = 40 * num_images
    max_per_set = 800 / (imdb.num_classes - 1) * num_images
    # heuristic: keep at most 100 detection per class per image prior to NMS
    max_per_image = 100
    
    # detection thresold for each class (this is adaptively set based on the
    # max_per_set constraint)
    thresh = -np.inf * np.ones(imdb.num_classes)
    # top_scores will hold one minheap of scores per class (used to enforce
    # the max_per_set constraint)
    top_scores = [[] for _ in xrange(imdb.num_classes)]
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]
    # number of boxes
    num_boxes = 0.0

    output_dir = get_output_dir(imdb, net['full'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_gt = 0.0
    num_det = 0.0

#    gt_roidb = imdb.gt_roidb()

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    for i in xrange(num_images):     
        if prop_boxes[i].shape[0] == 0:
            continue
        
        im = cv2.imread(imdb.image_path_at(i))
        _t['im_detect'].tic()
        scores, boxes = im_detect(net, im, prop_boxes[i], imdb.num_classes)
        num_boxes += scores.shape[0]
        _t['im_detect'].toc()
        
        # Evaluate the recall after bounding box regression
        
#        for j in xrange(1, imdb.num_classes):
            # For each class, evaluate the number of boxes         
#            inds = np.where(gt_roidb[i]['gt_classes'] == j)[0]
#            gt_boxes = gt_roidb[i]['boxes'][inds, :]
            
#            if gt_boxes.shape[0] > 0 and boxes.shape[0] > 0:
#                overlaps = bbox_overlaps(boxes[:, 4*j:4*j+4].astype(np.float), 
#                                         gt_boxes.astype(np.float))
                
#                det_inds = np.where(np.max(overlaps, axis=0) >= 0.5)[0] 
#                num_det += len(det_inds)
            
#            num_gt += gt_boxes.shape[0]

        _t['misc'].tic()
        for j in xrange(1, imdb.num_classes):            
            inds = np.where((scores[:, j] > thresh[j]))[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            
#            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
#                    .astype(np.float32, copy=False)
#            keep = nms(dets, cfg.TEST.NMS)
#            cls_scores = cls_scores[keep]
#            cls_boxes = cls_boxes[keep]
            
            top_inds = np.argsort(-cls_scores)[:max_per_image]
            cls_scores = cls_scores[top_inds]
            cls_boxes = cls_boxes[top_inds, :]
            
            # push new scores onto the minheap
            for val in cls_scores:
                heapq.heappush(top_scores[j], val)
            # if we've collected more than the max number of detection,
            # then pop items off the minheap and update the class threshold
            if len(top_scores[j]) > max_per_set:
                while len(top_scores[j]) > max_per_set:
                    heapq.heappop(top_scores[j])
                thresh[j] = top_scores[j][0]

            all_boxes[j][i] = \
                    np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                    .astype(np.float32, copy=False)

            if 0:
                keep = nms(all_boxes[j][i], 0.3)
                vis_detections(im, imdb.classes[j], all_boxes[j][i][keep, :])
        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)

    for j in xrange(1, imdb.num_classes):
        for i in xrange(num_images):
            if prop_boxes[i].shape[0] == 0:
                continue
            inds = np.where(all_boxes[j][i][:, -1] > thresh[j])[0]
            all_boxes[j][i] = all_boxes[j][i][inds, :]

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'Applying NMS to all detections'
    nms_dets = apply_nms(all_boxes, cfg.TEST.NMS)

    print 'Evaluating detections'
    imdb.evaluate_detections(nms_dets, output_dir)
    
    print 'The average time is proposal {:.3f}s, detection {:.3f}s'.\
        format(prop['time'], _t['im_detect'].average_time)
        
    print 'On average, {0} boxes per image are generated'.format(num_boxes/num_images)
    
#    print 'The raw recall is {:.3f}, the recall with bbox regression is {:.3f}'.format(prop['recall'], num_det/num_gt)
    
def test_net_shared(sc_net, frcnn_net, imdb):
    """Use shared convolutional layers for detection."""
    num_images = len(imdb.image_index)
    # heuristic: keep an average of 40 detections per class per images prior
    # to NMS
#    max_per_set = 40 * num_images
    max_per_set = 800 / (imdb.num_classes - 1) * num_images
    # heuristic: keep at most 100 detection per class per image prior to NMS
    max_per_image = 100
    # detection thresold for each class (this is adaptively set based on the
    # max_per_set constraint)
    thresh = -np.inf * np.ones(imdb.num_classes)
    # top_scores will hold one minheap of scores per class (used to enforce
    # the max_per_set constraint)
    top_scores = [[] for _ in xrange(imdb.num_classes)]
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]
    # number of boxes
    num_boxes = 0.0

    output_dir = get_output_dir(imdb, sc_net['full'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_gt = 0.0
    num_det = 0.0
    
#    gt_roidb = imdb.gt_roidb()

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    for i in xrange(num_images):
        
        im = cv2.imread(imdb.image_path_at(i))
        _t['im_detect'].tic()
        scores, boxes = im_detect_shared(sc_net, frcnn_net, im, imdb.num_classes)
        num_boxes += scores.shape[0]
        _t['im_detect'].toc()
        
        # Evaluate the recall after bounding box regression
        
#        for j in xrange(1, imdb.num_classes):
            # For each class, evaluate the number of boxes         
#            inds = np.where(gt_roidb[i]['gt_classes'] == j)[0]
#            gt_boxes = gt_roidb[i]['boxes'][inds, :]
            
#            if gt_boxes.shape[0] > 0 and boxes.shape[0] > 0:
#                overlaps = bbox_overlaps(boxes[:, 4*j:4*j+4].astype(np.float), 
#                                         gt_boxes.astype(np.float))
                
#                det_inds = np.where(np.max(overlaps, axis=0) >= 0.5)[0] 
#                num_det += len(det_inds)
                
#                num_gt += gt_boxes.shape[0]

        _t['misc'].tic()
        for j in xrange(1, imdb.num_classes):            
            inds = np.where((scores[:, j] > thresh[j]))[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            top_inds = np.argsort(-cls_scores)[:max_per_image]
            cls_scores = cls_scores[top_inds]
            cls_boxes = cls_boxes[top_inds, :]
            # push new scores onto the minheap
            for val in cls_scores:
                heapq.heappush(top_scores[j], val)
            # if we've collected more than the max number of detection,
            # then pop items off the minheap and update the class threshold
            if len(top_scores[j]) > max_per_set:
                while len(top_scores[j]) > max_per_set:
                    heapq.heappop(top_scores[j])
                thresh[j] = top_scores[j][0]

            all_boxes[j][i] = \
                    np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                    .astype(np.float32, copy=False)

            if 0:
                keep = nms(all_boxes[j][i], 0.3)
                vis_detections(im, imdb.classes[j], all_boxes[j][i][keep, :])
        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)

    for j in xrange(1, imdb.num_classes):
        for i in xrange(num_images):
            inds = np.where(all_boxes[j][i][:, -1] > thresh[j])[0]
            all_boxes[j][i] = all_boxes[j][i][inds, :]

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'Applying NMS to all detections'
    nms_dets = apply_nms(all_boxes, cfg.TEST.NMS)

    print 'Evaluating detections'
    imdb.evaluate_detections(nms_dets, output_dir)
    
    print 'The average detection time is {:.3f}s'.\
        format(_t['im_detect'].average_time)
        
    print 'On average, {0} boxes per image are proposed'.format(num_boxes/num_images)
    
#    print 'The recall is {:.3f}'.format(num_det/num_gt)
