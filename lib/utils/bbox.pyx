# --------------------------------------------------------
# Object detection using AZ-Net
# Written by Yongxi Lu
# Modified from Fast R-CNN
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Sergey Karayev
# --------------------------------------------------------
from numpy.lib.format import dtype_to_descr

cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def bbox_zoom_labels(
        np.ndarray[DTYPE_t, ndim=2] rois,
        np.ndarray[DTYPE_t, ndim=2] gt_rois,
        DTYPE_t max_area_ratio):
    """
    Compute information necessary for zoom labels
    """
    # overlaps = area(box1 and box2) / area(box2)
    # area_ratio = area(box1) / area(box 2)
    # for area_ratio, only entries in which overlaps>0 is valid
    cdef unsigned int N = rois.shape[0]
    cdef unsigned int K = gt_rois.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] overlaps = np.zeros((N, K), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] area_ratio = max_area_ratio * np.ones((N,K), dtype=DTYPE)
    
    cdef unsigned int k, n
    for k in range(K):
        gt_area = (
            (gt_rois[k, 2] - gt_rois[k, 0] + 1) *
            (gt_rois[k, 3] - gt_rois[k, 1] + 1)
        )
        for n in range(N):
            rois_area = float(
                (rois[n, 2] - rois[n, 0] + 1) *
                (rois[n, 3] - rois[n, 1] + 1)
            )
            area_ratio[n, k] = gt_area / (rois_area + 1e-14)
            if area_ratio[n, k] <= max_area_ratio:
                iw = (
                    min(rois[n, 2], gt_rois[k, 2]) -
                    max(rois[n, 0], gt_rois[k, 0]) + 1
                )
                if iw > 0:
                    ih = (
                        min(rois[n, 3], gt_rois[k, 3]) -
                        max(rois[n, 1], gt_rois[k, 1]) + 1
                    )
                    if ih > 0:
                        overlaps[n, k] = iw * ih / (gt_area + 1e-14)
    
    return overlaps, area_ratio

def bbox_coverage(
        np.ndarray[DTYPE_t, ndim=2] boxes,
        np.ndarray[DTYPE_t, ndim=2] query_boxes,
		DTYPE_t tolerance=0.0):
    """
    Find out query_boxes covered by boxes, excluding boxes that are too large.  
    ----------
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
	tolerance: scalar of float
    Returns
    -------
    coverage: (N, K) ndarray of coverage scores, same as iou
	is_cover: (N, K) ndarray logical array of existence of valid coverage
	cover_boxes: (N, 4) ndarray of extended covering boxes
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] coverage = np.zeros((N, K), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] cover_boxes = boxes.copy()

    is_cover = np.zeros((N,K), dtype=np.bool)
    cdef np.ndarray[DTYPE_t, ndim=1] ua = np.empty((N,), dtype=DTYPE)
    cdef DTYPE_t itop, ileft, ibottom, iright
    cdef DTYPE_t mtop, mleft, mbottom, mright
    
    cdef DTYPE_t width, height
    cdef DTYPE_t w_t, h_t
    
    for n in range(N):
        mtop, mleft, mbottom, mright = 0.0, 0.0, 0.0, 0.0
        width, height = boxes[n,2]-boxes[n,0]+1, boxes[n,3]-boxes[n,1]+1
        w_t, h_t = np.floor(width*tolerance), np.floor(height*tolerance)
        
        for k in range(K):		
            ileft = max(0, boxes[n,0] - query_boxes[k,0])
            itop = max(0, boxes[n,1] - query_boxes[k,1])
            iright = max(0, query_boxes[k,2] - boxes[n,2])
            ibottom = max(0, query_boxes[k,3] - boxes[n,3])
            
            if ileft<=w_t and itop <= h_t and ibottom <= h_t and iright <= w_t:
                mtop = max(mtop, itop)
                mleft = max(mleft, ileft)
                mbottom = max(mbottom, ibottom)
                mright = max(mright, iright) 
                is_cover[n,k] = True

        cover_boxes[n,0] -= mleft
        cover_boxes[n,1] -= mtop
        cover_boxes[n,2] += mright
        cover_boxes[n,3] += mbottom
        ua[n] = (
		    (cover_boxes[n,2] - cover_boxes[n,0] + 1) *
             (cover_boxes[n,3] - cover_boxes[n,1] + 1)
            )		 

    cdef DTYPE_t iw, ih, box_area    
    for k in range(K):
        box_area = (
            (query_boxes[k,2] - query_boxes[k,0] + 1) *
            (query_boxes[k,3] - query_boxes[k,1] + 1)
        )
        for n in range(N):
            if is_cover[n,k] == True:
                coverage[n,k] = box_area / ua[n]

    return coverage, is_cover, cover_boxes	

def bbox_overlaps(
        np.ndarray[DTYPE_t, ndim=2] boxes,
        np.ndarray[DTYPE_t, ndim=2] query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] overlaps = np.zeros((N, K), dtype=DTYPE)
    cdef DTYPE_t iw, ih, box_area
    cdef DTYPE_t ua
    cdef unsigned int k, n
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = float(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps
