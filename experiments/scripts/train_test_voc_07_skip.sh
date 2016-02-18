#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/train_test_voc_skip.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_az_net.py --gpu $1 \
  --solver models/Pascal/VGG16_skip/az-net/solver.prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --imdb voc_2007_trainval \
  --cfg experiments/cfgs/vgg16_voc2007_skip_train.yml \
  --iters 160000

time ./tools/train_det_net.py --gpu $1 \
  --solver models/Pascal/VGG16_skip/frcnn/solver.prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --def models/Pascal/VGG16_skip/az-net/test.prototxt \
  --def_fc models/Pascal/VGG16_skip/az-net/test_fc.prototxt \
  --net output/az-net/voc_2007_trainval/vgg16_az_net_skip_iter_160000.caffemodel \
  --imdb voc_2007_trainval \
  --cfg experiments/cfgs/vgg16_voc2007_skip_train.yml \
  --iters 80000

time ./tools/train_az_net.py --gpu $1 \
  --solver models/Pascal/VGG16_skip/az-net/shared/solver.prototxt \
  --weights output/az-net/voc_2007_trainval/vgg16_fast_rcnn_skip_iter_80000.caffemodel \
  --imdb voc_2007_trainval \
  --cfg experiments/cfgs/vgg16_voc2007_skip_train.yml \
  --iters 160000

time ./tools/train_det_net.py --gpu $1 \
  --solver models/Pascal/VGG16_skip/frcnn/shared/solver.prototxt \
  --weights output/az-net/voc_2007_trainval/vgg16_fast_rcnn_skip_iter_80000.caffemodel \
  --def models/Pascal/VGG16_skip/az-net/test.prototxt \
  --def_fc models/Pascal/VGG16_skip/az-net/test_fc.prototxt \
  --net output/az-net/voc_2007_trainval/vgg16_az_net_skip_shared_iter_160000.caffemodel \
  --imdb voc_2007_trainval \
  --cfg experiments/cfgs/vgg16_voc2007_skip_train.yml \
  --iters 80000

time ./tools/test_shared.py --gpu $1 \
  --def_fc_frcnn models/Pascal/VGG16_skip/frcnn/test_fc.prototxt \
  --net_frcnn output/az-net/voc_2007_trainval/vgg16_fast_rcnn_skip_shared_iter_80000.caffemodel \
  --def_az models/Pascal/VGG16_skip/az-net/test.prototxt \
  --def_fc_az models/Pascal/VGG16_skip/az-net/test_fc.prototxt \
  --net_az output/az-net/voc_2007_trainval/vgg16_az_net_skip_shared_iter_160000.caffemodel \
  --cfg experiments/cfgs/vgg16_voc2007_skip.yml \
  --imdb voc_2007_test \
  --comp
