#!/bin/bash

# usage: exp_voc_shared gpu-id config-file train-set test-set
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

LOG="experiments/logs/voc_shared.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
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

time ./tools/train_az_net.py --gpu $gpu_id \
  --solver models/Pascal/VGG16/az-net/shared/solver.prototxt \
  --weights output/az-net/voc_2007_trainval/vgg16_fast_rcnn_iter_80000.caffemodel \
  --imdb $trainset \
  --cfg experiments/cfgs/$cfg_file \
  --iters 160000

time ./tools/train_det_net.py --gpu $gpu_id \
  --solver models/Pascal/VGG16/frcnn/shared/solver.prototxt \
  --weights output/az-net/voc_2007_trainval/vgg16_fast_rcnn_iter_80000.caffemodel \
  --def models/Pascal/VGG16/az-net/test.prototxt \
  --def_fc models/Pascal/VGG16/az-net/test_fc.prototxt \
  --net output/az-net/voc_2007_trainval/vgg16_az_net_shared_iter_160000.caffemodel \
  --imdb $trainset \
  --cfg experiments/cfgs/$cfg_file \
  --iters 80000

time ./tools/test_shared.py --gpu $gpu_id \
  --def_fc_frcnn models/Pascal/VGG16/frcnn/test_fc.prototxt \
  --net_frcnn output/az-net/voc_2007_trainval/vgg16_fast_rcnn_shared_iter_80000.caffemodel \
  --def_az models/Pascal/VGG16/az-net/test.prototxt \
  --def_fc_az models/Pascal/VGG16/az-net/test_fc.prototxt \
  --net_az output/az-net/voc_2007_trainval/vgg16_az_net_shared_iter_160000.caffemodel \
  --cfg experiments/cfgs/$cfg_file \
  --imdb $testset \
  --comp
