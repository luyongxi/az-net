#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/train_test_unshared_coco.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_sc_net.py --gpu $1 \
  --solver models/COCO/VGG16/az-net/solver.prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --imdb coco_trainval_2014 \
  --cfg experiments/cfgs/vgg16_coco_train.yml \
  --iters 720000

time ./tools/train_det_net.py --gpu $1 \
  --solver models/COCO/VGG16/frcnn/solver.prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --def models/COCO/VGG16/az-net/test.prototxt \
  --def_fc models/COCO/VGG16/az-net/test_fc.prototxt \
  --net output/az-net/coco_2014_trainval/vgg16_az_net_iter_720000.caffemodel \
  --imdb coco_trainval_2014 \
  --cfg experiments/cfgs/vgg16_coco_train.yml \
  --iters 720000

time ./tools/prop_az.py --gpu $1 \
  --def models/COCO/VGG16/az-net/test.prototxt \
  --def_fc models/COCO/VGG16/az-net/test_fc.prototxt \
  --net output/az-net/coco_2014_trainval/vgg16_az_net_iter_720000.caffemodel \
  --imdb coco_test-dev_2015 \
  --cfg experiments/cfgs/vgg16_coco.yml

time ./tools/test_det_net.py --gpu $1 \
  --def models/COCO/VGG16/frcnn/test.prototxt \
  --net output/az-net/coco_2014_trainval/vgg16_fast_rcnn_iter_720000.caffemodel \
  --prop output/az-net/coco_2015_test-dev/vgg16_az_net_iter_720000/proposals.pkl \
  --imdb coco_test-dev_2015 \
  --cfg experiments/cfgs/vgg16_coco.yml
