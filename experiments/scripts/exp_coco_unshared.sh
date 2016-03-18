#!/bin/bash

# usage: exp_coco_unshared gpu-id config-file prefix train-set test-set
# default prefix: default
# default train set: coco_trainval_2014
# default test set: coco_test-dev_2015

gpu_id=$1
cfg_file=$2

prefix="default"
trainset="coco_trainval_2014"
testset="coco_test-dev_2015"

if [ $# -eq 0 ]; then
  echo Usage: exp_coco_unshared gpu-id config-file prefix train-set test-set
  exit 1
fi

if [ $# -eq 5 ] ; then
  prefix=$3
  trainset=$4
  testset=$5
elif [ $# -eq 4 ] ; then
  prefix=$3
  trainset=$4
elif [ $# -eq 3 ] ; then
  prefix=$3
fi

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/coco_unshared.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Using cfg file "$cfg_file"
echo Using prefix "$prefix"
echo Training set "$trainset", test set "$testset"
echo Logging output to "$LOG"

time ./tools/train_az_net.py --gpu $gpu_id \
  --solver models/COCO/VGG16/az-net/solver_"$prefix".prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --imdb $trainset \
  --cfg experiments/cfgs/$cfg_file \
  --iters 720000 \
  --exp $prefix

time ./tools/train_det_net.py --gpu $gpu_id \
  --solver models/COCO/VGG16/frcnn/solver_"$prefix".prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --def models/COCO/VGG16/az-net/test.prototxt \
  --def_fc models/COCO/VGG16/az-net/test_fc.prototxt \
  --net output/$prefix/$trainset/vgg16_az_net_iter_720000.caffemodel \
  --imdb $trainset \
  --cfg experiments/cfgs/$cfg_file \
  --iters 720000 \
  --exp $prefix

time ./tools/set_thresh.py --gpu $gpu_id \
  --def models/Pascal/VGG16/az-net/test.prototxt \
  --def_fc models/Pascal/VGG16/az-net/test_fc.prototxt \
  --net output/$prefix/$trainset/vgg16_az_net_iter_160000.caffemodel \
  --imdb $trainset \
  --cfg experiments/cfgs/$cfg_file \
  --exp $prefix

time ./tools/prop_az.py --gpu $gpu_id \
  --def models/COCO/VGG16/az-net/test.prototxt \
  --def_fc models/COCO/VGG16/az-net/test_fc.prototxt \
  --net output/$prefix/$trainset/vgg16_az_net_iter_720000.caffemodel \
  --imdb $testset \
  --thresh output/$prefix/$trainset/vgg16_az_net_iter_720000/thresh.pkl \
  --cfg experiments/cfgs/$cfg_file \
  --exp $prefix

time ./tools/test_det_net.py --gpu $gpu_id \
  --def models/COCO/VGG16/frcnn/test.prototxt \
  --net output/$prefix/$trainset/vgg16_fast_rcnn_iter_720000.caffemodel \
  --prop output/$prefix/$testset/vgg16_az_net_iter_720000/proposals.pkl \
  --imdb $testset \
  --cfg experiments/cfgs/$cfg_file \
  --exp $prefix
