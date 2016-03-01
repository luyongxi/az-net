# --------------------------------------------------------
# Object detection using AZ-Net
# Written by Yongxi Lu
# Modified from Fast R-CNN
# --------------------------------------------------------

"""Functions designed for fine-grained analysis of region proposals.
   All functions are designed with the assumption that the input RoIs
   has labels, so can be analyzed in a fine-grained manner. 
"""

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
import scipy.io as sio

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
    conv_name = cfg.SEAR.AZ_CONV
    
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
                                            blobs = conv_name)
            conv = {name: blobs_out[name] for name in conv_name}
        else:
            for name in conv_name:
                net['fc'].blobs[name].reshape(*(conv[name].shape))
            net['fc'].blobs['rois'].reshape(*(blobs['rois'].shape))
            blobs_out = net['fc'].forward(rois=blobs['rois'].astype(np.float32, copy=False),
                                          **(conv))
            
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

def im_propose(net, im):
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
    Z = B
    # anchor region history
    Bhis = np.zeros((0,5))
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
    Tz = 0
#    step = cfg.SEAR.Tz/K
    # cached convolutional layer (the last layer)
    conv = None
    for k in xrange(K):
        # Get zoom indicator and adjacent predictions (w/ confidence)
        # Scores and aBBox already sifted and vectorized by sc_net function
#        print B.shape[0]
        # anchor region history
        Bhis = np.vstack((Bhis, np.hstack((B, k * np.ones((B.shape[0],1))))))
        zoom, boxes, c, conv = _az_forward(net, im, B, conv)
        num_eval = num_eval + B.shape[0]
        # predictions with high confidence scores are included
        Y = np.vstack((Y, boxes))
        aScores = np.hstack((aScores, c))
        # Z is updated to regions for zoom in
        indZ = np.where(zoom >= Tz)[0]
        Z = B[indZ, :]
        if Z.shape[0] == 0:
            break
        # B is updated to be regions that are expanded from it
        B = divide_region(Z)
        
#        Tz = Tz + step
        Tz = cfg.SEAR.Tz
        
    num_proposals = cfg.SEAR.NUM_PROPOSALS
    indA = np.argsort(-aScores)
    max_num = np.minimum(num_proposals, Y.shape[0])
    Y = Y[indA[:max_num], :]
            
    print '{0} proposals, evaluate {1} regions, reaches depth {2}.'\
        .format(Y.shape[0], num_eval,k)

    return np.hstack((Y, aScores[indA[:max_num, np.newaxis]])), Bhis

def test_proposals(net, imdb):
    """The purpose of this function is to record all information necessary for
    fine-grained analysis: including ground truths, proposals, and all anchor regions..."""
    num_images = len(imdb.image_index)
    # all proposals are collected into:
    #    prop_boxes[cls] = N x 5 array of proposals in
    #    (x1, y1, x2, y2, score)
    prop_boxes = np.zeros((num_images,), dtype=np.object)
    anchor_boxes = np.zeros((num_images,), dtype=np.object)
    gt_boxes = np.zeros((num_images,), dtype=np.object)
    ss_boxes = np.zeros((num_images,), dtype=np.object)
    fn = np.zeros((num_images,), dtype=np.object)
    im_shapes = np.zeros((num_images,), dtype=np.object)

    output_dir = get_output_dir(imdb, net['full'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # timers
    _t = {'im_prop' : Timer()}
    
    gt_roidb = imdb.gt_roidb()
    roidb = imdb.roidb

    for i in xrange(num_images):
        im = cv2.imread(imdb.image_path_at(i))
        im_shapes[i] = im.shape       
        _t['im_prop'].tic()
        prop_boxes[i], anchor_boxes[i] = \
            im_propose(net, im) 
        
        _t['im_prop'].toc()
    
        gt_boxes[i] = gt_roidb[i]['boxes']
        fn[i] = imdb.image_path_at(i)
        
        gt_overlaps = roidb[i]['gt_overlaps'].toarray()        
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # find out ground truths
        ss_inds = np.where(max_overlaps < 1)[0]
        ss_boxes[i] = roidb[i]['boxes'][ss_inds, :]

        print 'im_detect: {:d}/{:d} {:.3f}s' \
              .format(i + 1, num_images, _t['im_prop'].average_time)
        
    Tz = cfg.SEAR.Tz
    num_proposals = cfg.SEAR.NUM_PROPOSALS
        
    outfile = os.path.join(output_dir, 'AZ_results.mat')
    sio.savemat(outfile, dict(prop_boxes = prop_boxes, 
                              anchor_boxes = anchor_boxes,
                              gt_boxes = gt_boxes,
                              fn = fn,
                              Tz = Tz,
                              num_proposals = num_proposals,
                              im_shapes = im_shapes,
                              ss_boxes = ss_boxes))
