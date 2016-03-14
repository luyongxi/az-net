#!/bin/bash

# usage: exp_coco_unshared gpu-id config-file train-set test-set
# default train set: coco_trainval_2014
# default test set: coco_test-dev_2015

gpu_id=$1
cfg_file=$2

if ["$#" = 3] ; then
  trainset=$3
  testset=$4
else
  trainset="coco_trainval_2014"
  testset="coco_test-dev_2015" 
fi

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/coco_unshared.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Using cfg file "$cfg_file"
echo Training set "$trainset", test set "$testset"
echo Logging output to "$LOG"

time ./tools/train_sc_net.py --gpu $gpu_id \
  --solver models/COCO/VGG16/az-net/solver.prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --imdb $trainset \
  --cfg experiments/cfgs/$cfg_file \
  --iters 720000

time ./tools/train_det_net.py --gpu $gpu_id \
  --solver models/COCO/VGG16/frcnn/solver.prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --def models/COCO/VGG16/az-net/test.prototxt \
  --def_fc models/COCO/VGG16/az-net/test_fc.prototxt \
  --net output/az-net/coco_2014_trainval/vgg16_az_net_iter_720000.caffemodel \
  --imdb $trainset \
  --cfg experiments/cfgs/$cfg_file \
  --iters 720000

time ./tools/prop_az.py --gpu $gpu_id \
  --def models/COCO/VGG16/az-net/test.prototxt \
  --def_fc models/COCO/VGG16/az-net/test_fc.prototxt \
  --net output/az-net/coco_2014_trainval/vgg16_az_net_iter_720000.caffemodel \
  --imdb $testset \
  --cfg experiments/cfgs/$cfg_file

time ./tools/test_det_net.py --gpu $gpu_id \
  --def models/COCO/VGG16/frcnn/test.prototxt \
  --net output/az-net/coco_2014_trainval/vgg16_fast_rcnn_iter_720000.caffemodel \
  --prop output/az-net/coco_2015_test-dev/vgg16_az_net_iter_720000/proposals.pkl \
  --imdb $testset \
  --cfg experiments/cfgs/$cfg_file
