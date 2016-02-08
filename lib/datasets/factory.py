# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import datasets.pascal_voc
import datasets.sun_merge
import datasets.coco
import numpy as np

# Set up COCO 2014 dataset
for year in ['2014']:
    for split in ['train','val','test','trainval']:
        name = 'coco_{}_{}'.format(split, year)
        __sets[name] = (lambda split=split, year=year:
                datasets.coco(split, year))

# Set up COCO 2015 dataset
for year in ['2015']:
    for split in ['test','test-dev']:
        name = 'coco_{}_{}'.format(split, year)
        __sets[name] = (lambda split=split, year=year:
                datasets.coco(split, year))

# Set up voc_<year>_<split>
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year:
                datasets.pascal_voc(split, year))

for year in ['07+12']:
    for split in ['trainval']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year:
                datasets.pascal_voc(split, year))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
