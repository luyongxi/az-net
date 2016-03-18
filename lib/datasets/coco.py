# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import datasets.coco
import os
import datasets.imdb
import numpy as np
import scipy.sparse
import scipy.io as sio
import cPickle
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

class coco(datasets.imdb):
    def __init__(self, image_set, year, devkit_path=None):
        datasets.imdb.__init__(self, 'coco_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'images')
        
        # initialize COCO API
        if image_set == 'trainval':
            self._annFile = [[None] for _ in xrange(2)]
            self._coco = [[] for _ in xrange(2)]
            self._annFile[0] = os.path.join(self._devkit_path, 'annotations', 
                             'instances_' + 'train' + year + '.json')
            self._annFile[1] = os.path.join(self._devkit_path, 'annotations', 
                             'instances_' + 'val' + year + '.json')
            self._coco[0] = COCO(self._annFile[0])
            self._coco[1] = COCO(self._annFile[1])
        elif image_set == 'train' or image_set == 'val':
            self._annFile = [[None] for _ in xrange(1)]
            self._coco = [[] for _ in xrange(1)]
            self._annFile[0] = os.path.join(self._devkit_path, 'annotations', 
                                         'instances_' + image_set + year + '.json')
            self._coco[0] = COCO(self._annFile[0])
        else:
            self._annFile = [[None] for _ in xrange(1)]
            self._coco = [[] for _ in xrange(1)]
            self._annFile[0] = os.path.join(self._devkit_path, 'annotations', 
                                         'image_info_' + image_set + year + '.json')
            self._coco[0] = COCO(self._annFile[0])
        
        self._classes = (u'__background__', u'person', u'bicycle', u'car', 
                         u'motorcycle', u'airplane', u'bus', u'train', u'truck',
                         u'boat', u'traffic light', u'fire hydrant', u'stop sign',
                         u'parking meter', u'bench', u'bird', u'cat', u'dog', 
                         u'horse', u'sheep', u'cow', u'elephant', u'bear', u'zebra',
                         u'giraffe', u'backpack', u'umbrella', u'handbag', u'tie',
                         u'suitcase', u'frisbee', u'skis', u'snowboard', u'sports ball',
                         u'kite', u'baseball bat', u'baseball glove', u'skateboard',
                         u'surfboard', u'tennis racket', u'bottle', u'wine glass',
                         u'cup', u'fork', u'knife', u'spoon', u'bowl', u'banana', u'apple',
                         u'sandwich', u'orange', u'broccoli', u'carrot', u'hot dog',
                         u'pizza', u'donut', u'cake', u'chair', u'couch', u'potted plant',
                         u'bed', u'dining table', u'toilet', u'tv', u'laptop', u'mouse',
                         u'remote', u'keyboard', u'cell phone', u'microwave', u'oven',
                         u'toaster', u'sink', u'refrigerator', u'book', u'clock', u'vase',
                         u'scissors', u'teddy bear', u'hair drier', u'toothbrush') 
        
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_index, self._set_index = self._load_image_set_index()
        
        # Default to roidb handler
        self._roidb_handler = self.load_roidb

        assert os.path.exists(self._devkit_path), \
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """        
        i = self._image_index.index(index)
        setId = self._set_index[i]
        
        image_path = os.path.join(self._data_path, 
                                  self._coco[setId].loadImgs(index)[0]['file_name'])
        
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        imgIdx = []
        setIdx = []
        for i in xrange(len(self._coco)):
            thisIdx = self._coco[i].getImgIds()
            imgIdx = imgIdx + thisIdx
            setIdx = setIdx + [i for _ in xrange(len(thisIdx))]
        
        return imgIdx, setIdx

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'COCO')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        """
        gt_roidb = [self._load_coco_annotation(index)
                    for index in self.image_index]
        
        return gt_roidb
    
    def load_roidb(self):
        """
        Return the database of ground-truth ROIs
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb
        
        if self._image_set == 'train' or self._image_set == 'val' or self._image_set == 'trainval':
            roidb = self.gt_roidb()
        else:
            roidb = None
                
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)
          
        return roidb

    def _load_coco_annotation(self, index):
        """
        Load image and bounding boxes info from COCO API
        """
        setId = self._set_index[self._image_index.index(index)]
        annIds = self._coco[setId].getAnnIds(imgIds = index)
        anns = self._coco[setId].loadAnns(annIds)
        num_objs = len(anns)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        
        img = self._coco[setId].loadImgs(index)[0]
        height = img['height']
        width = img['width']
        
        # Load object bounding boxes
        for ix in xrange(num_objs):
            bbox = anns[ix]['bbox']
            # from (x,y,w,h) format to (x1,y1,x2,y2) format
            x1 = min(width-1.0, max(0.0, float(bbox[0])))
            y1 = min(height-1.0, max(0.0, float(bbox[1])))
            x2 = min(width-1.0, x1 + max(0.0, float(bbox[2])))
            y2 = min(height-1.0, y1 + max(0.0, float(bbox[3])))
            
            # object class
            category_id = anns[ix]['category_id']
            cls = self._coco[0].getCatIds().index(category_id) + 1
            
            # save
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}

    def _write_coco_results_file(self, all_boxes, output_dir):
        """
        Save results in COCO format
        """
        
        catIds = self._coco[0].getCatIds()
        dets = []
        for cls_ind in xrange(1, len(self.classes)):

            for im_ind, index in enumerate(self.image_index):
                cls_dets = all_boxes[cls_ind][im_ind]
                
                if cls_dets != []:
                    for k in xrange(cls_dets.shape[0]):
                        x = float(cls_dets[k, 0])
                        y = float(cls_dets[k, 1])
                        width = float(cls_dets[k, 2] - cls_dets[k, 0] + 1.0)
                        height = float(cls_dets[k, 3] - cls_dets[k, 1] + 1.0)
                        score = float(cls_dets[k, -1])
                        
                        x = int(x * 100) / 100.0
                        y = int(y * 100) / 100.0
                        width = int(width * 100) / 100.0
                        height = int(height * 100) / 100.0

                        dets.append({"image_id": index, "category_id": catIds[cls_ind-1], 
                                         "bbox": [x, y, width, height], 
                                         "score": score})
        # save in COCO json format
        filename = os.path.join(output_dir, 'instances_' + 
                                self._image_set + self._year +
                                '_results.json')
        with open(filename, 'wt') as f:
            json.dump(dets, f)
#            f.write(packed_dets)
        
        return filename
    
    def write_coco_multiple_files(self, all_boxes, size, output_dir):
        """
        Save results in COCO format, to multiple files
        """
        
        catIds = self._coco[0].getCatIds()
        dets = []
        
        split_id = 0
                
        for im_ind, index in enumerate(self.image_index):
            for cls_ind in xrange(1, len(self.classes)):

                cls_dets = all_boxes[cls_ind][im_ind]
                
                if cls_dets != []:
                    for k in xrange(cls_dets.shape[0]):
                        x = float(cls_dets[k, 0])
                        y = float(cls_dets[k, 1])
                        width = float(cls_dets[k, 2] - cls_dets[k, 0] + 1.0)
                        height = float(cls_dets[k, 3] - cls_dets[k, 1] + 1.0)
                        score = float(cls_dets[k, -1])
                        
                        x = int(x * 100) / 100.0
                        y = int(y * 100) / 100.0
                        width = int(width * 100) / 100.0
                        height = int(height * 100) / 100.0

                        dets.append({"image_id": index, "category_id": catIds[cls_ind-1], 
                                         "bbox": [x, y, width, height], 
                                         "score": score})
            if (im_ind + 1) % size == 0 or im_ind == len(self.image_index) - 1:
                
                # save in COCO json format
                filename = os.path.join(output_dir, 'instances_' + 
                                        self._image_set + self._year +
                                        '_results_' + str(split_id) + '.json')
                with open(filename, 'wt') as f:
                    json.dump(dets, f)
                
                split_id = split_id + 1
                dets = []

    def _do_coco_eval(self, dtFile, output_dir):
        """
        Evaluate using COCO API
        """
        if self._image_set == 'train' or self._image_set == 'val':
            cocoGt = self._coco[0]
            cocoDt = COCO(dtFile)
            E = COCOeval(cocoGt, cocoDt)
            E.evaluate()
            E.accumulate()
            E.summarize()
            
    def evaluate_detections(self, all_boxes, output_dir):
        if self._image_set != 'trainval':
            dtFile = self._write_coco_results_file(all_boxes, output_dir)
            self._do_coco_eval(dtFile, output_dir)

    def competition_mode(self, on):
        pass
    
    def append_flipped_images(self):
        num_images = self.num_images
        widths = [self._coco[self._set_index[i]].
                  loadImgs(self._image_index[i])[0]['width']
                  for i in xrange(num_images)]
        
        for i in xrange(num_images):
            boxes = self.roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2 - 1.0
            boxes[:, 2] = widths[i] - oldx1 - 1.0          
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {'boxes' : boxes,
                     'gt_overlaps' : self.roidb[i]['gt_overlaps'],
                     'gt_classes' : self.roidb[i]['gt_classes'],
                     'flipped' : True}
            self.roidb.append(entry)
        
        self._image_index = self._image_index * 2
        self._set_index = self._set_index * 2
        
    def image_size(self, i):
        img = self._coco[self._set_index[i]].\
                  loadImgs(self._image_index[i])[0]
        return (img['height'],img['width'])
