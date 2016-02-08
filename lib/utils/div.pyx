# --------------------------------------------------------
# Object detection using AZ-Net
# Written by Yongxi Lu
# Modified from Fast R-CNN
# --------------------------------------------------------
from numpy.lib.format import dtype_to_descr

cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def divide_region(np.ndarray[DTYPE_t, ndim=2] regions,
                  DTYPE_t min_height):
    """ Divide the regions into non-overlapping sub-regions
        The algorithm first finds the shorter side of a region,
        it then divides the image along that axis into 2 parts.
        Then it finds the closest division along the longer axis
        so that the generated regions are close to squares
    """    
    cdef unsigned int num_regions = regions.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] regions_out = np.zeros((0,4), dtype=DTYPE)
    cdef unsigned int min_ind, max_ind
    cdef unsigned int num_short, num_long, num_blocks
    cdef DTYPE_t l_short, l_long
    cdef np.ndarray[DTYPE_t, ndim=2] subregion = np.zeros((0,4), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] lengths = np.zeros((2,), dtype=DTYPE)
    
    for i in xrange(num_regions):
        lengths[0] = regions[i,2] - regions[i,0] + 1.0
        lengths[1] = regions[i,3] - regions[i,1] + 1.0
        
        min_ind = np.argmin(lengths)
        max_ind = 1 - min_ind
        
        # number of sub-regions along each axis and their length
        num_short = 2        
        l_short = lengths[min_ind] / 2
        # number of sub-regions along the longer side is naturally longer
        num_long = int(lengths[max_ind] / l_short)
        l_long = lengths[max_ind] / num_long
        
        num_blocks = num_short * num_long + (num_short - 1) * (num_long - 1)
        subregion = np.zeros((num_blocks,4))
        for k in xrange(num_short):
            for j in xrange(num_long):
                if min_ind == 0:    # width
                    subregion[k*num_long+j,:] = \
                            np.array([[k*l_short, j*l_long,
                                       (k+1)*l_short, (j+1)*l_long]])
                else:   # height
                    subregion[k*num_long+j,:] = \
                            np.array([[j*l_long, k*l_short, 
                                       (j+1)*l_long, (k+1)*l_short]])
        offset = num_short * num_long
        h_short = l_short/2
        h_long = l_long/2
        for k in xrange(num_short-1):
            for j in xrange(num_long-1):
                if min_ind == 0:    # width
                    subregion[k*num_long+j+offset,:] = \
                            np.array([[k*l_short+h_short, j*l_long+h_long,
                                       (k+1)*l_short+h_short, (j+1)*l_long+h_long]])
                else:   # height
                    subregion[k*num_long+j+offset,:] = \
                            np.array([[j*l_long+h_long, k*l_short+h_short, 
                                       (j+1)*l_long+h_long, (k+1)*l_short+h_short]])                 
        
        subregion[:,[0,2]] += regions[i,0]
        subregion[:,[1,3]] += regions[i,1]                        
        
        regions_out = np.vstack((regions_out, subregion))
        
    return _sift_dup(regions_out, min_height)

def _sift_dup(np.ndarray[DTYPE_t, ndim=2] regions,
              DTYPE_t min_height):

    """ Sift away duplicate regions
        A region is considered a duplicate it after down-sampling by 
        minimum height it became identical to another region
    """
    cdef np.ndarray[DTYPE_t, ndim=1] v = np.array([1, 1e3, 1e6, 1e9], dtype=DTYPE)
    hashes = np.round(regions / min_height).dot(v)
    _, index = np.unique(hashes, return_index=True)
    
    return regions[index, :]