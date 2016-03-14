#!/bin/bash

# usage: exp_voc_unshared gpu-id config-file train-set test-set
# default train set: voc_2007_trainval
# default test set: voc_2007_test

gpu_id=$1
cfg_file=$2

if ["$#" = 3] ; then
  trainset=$3
  testset=$4
else
  trainset="voc_2007_trainval"
  testset="voc_2007_test"
fi

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/voc_unshared.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Using cfg file "$cfg_file"
echo Training set "$trainset", test set "$testset"
echo Logging output to "$LOG"

time ./tools/train_az_net.py --gpu $gpu_id \
  --solver models/Pascal/VGG16/az-net/solver.prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --imdb $trainset \
  --cfg experiments/cfgs/$cfg_file \
  --iters 160000

time ./tools/train_det_net.py --gpu $gpu_id \
  --solver models/Pascal/VGG16/frcnn/solver.prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --def models/Pascal/VGG16/az-net/test.prototxt \
  --def_fc models/Pascal/VGG16/az-net/test_fc.prototxt \
  --net output/az-net/voc_2007_trainval/vgg16_az_net_iter_160000.caffemodel \
  --imdb $trainset \
  --cfg experiments/cfgs/$cfg_file \
  --iters 80000

time ./tools/prop_az.py --gpu $gpu_id \
  --def models/Pascal/VGG16/az-net/test.prototxt \
  --def_fc models/Pascal/VGG16/az-net/test_fc.prototxt \
  --net output/az-net/voc_2007_trainval/vgg16_az_net_iter_160000.caffemodel \
  --imdb $testset \
  --cfg experiments/cfgs/$cfg_file

time ./tools/test_det_net.py --gpu $gpu_id \
  --def models/Pascal/VGG16/frcnn/test.prototxt \
  --net output/az-net/voc_2007_trainval/vgg16_fast_rcnn_iter_80000.caffemodel \
  --prop output/az-net/voc_2007_test/vgg16_az_net_iter_160000/proposals.pkl \
  --imdb $testset \
  --cfg experiments/cfgs/$cfg_file \
  --comp
