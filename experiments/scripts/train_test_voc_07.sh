#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/train_test_voc.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_az_net.py --gpu $1 \
  --solver models/Pascal/VGG16/az-net/solver.prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --imdb voc_2007_trainval \
  --cfg experiments/cfgs/vgg16_voc2007_train.yml \
  --iters 80000

time ./tools/train_det_net.py --gpu $1 \
  --solver models/Pascal/VGG16/frcnn/solver.prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --def models/Pascal/VGG16/az-net/test.prototxt \
  --def_fc models/Pascal/VGG16/az-net/test_fc.prototxt \
  --net output/az-net/voc_2007_trainval/vgg16_az_net_iter_80000.caffemodel \
  --imdb voc_2007_trainval \
  --cfg experiments/cfgs/vgg16_voc2007_train.yml \
  --iters 80000

time ./tools/train_az_net.py --gpu $1 \
  --solver models/Pascal/VGG16/az-net/shared/solver.prototxt \
  --weights output/az-net/voc_2007_trainval/vgg16_fast_rcnn_iter_80000.caffemodel \
  --imdb voc_2007_trainval \
  --cfg experiments/cfgs/vgg16_voc2007_train.yml \
  --iters 80000

time ./tools/train_det_net.py --gpu $1 \
  --solver models/Pascal/VGG16/frcnn/shared/solver.prototxt \
  --weights output/az-net/voc_2007_trainval/vgg16_fast_rcnn_iter_80000.caffemodel \
  --def models/Pascal/VGG16/az-net/test.prototxt \
  --def_fc models/Pascal/VGG16/az-net/test_fc.prototxt \
  --net output/az-net/voc_2007_trainval/vgg16_az_net_shared_iter_80000.caffemodel \
  --imdb voc_2007_trainval \
  --cfg experiments/cfgs/vgg16_voc2007_train.yml \
  --iters 80000

time ./tools/test_shared.py --gpu $1 \
  --def_fc_frcnn models/Pascal/VGG16/frcnn/test_fc.prototxt \
  --net_frcnn output/az-net/voc_2007_trainval/vgg16_fast_rcnn_shared_iter_80000.caffemodel \
  --def_az models/Pascal/VGG16/az-net/test.prototxt \
  --def_fc_az models/Pascal/VGG16/az-net/test_fc.prototxt \
  --net_az output/az-net/voc_2007_trainval/vgg16_az_net_shared_iter_80000.caffemodel \
  --cfg experiments/cfgs/vgg16_voc2007.yml \
  --imdb voc_2007_test \
  --comp
