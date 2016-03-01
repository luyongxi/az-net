#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

# usage: ./train_test_voc_skip.sh gpu-id trainset testset

LOG="experiments/logs/train$2_test$3_voc_skip.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# az-net: finetune conv3-conv5
time ./tools/train_az_net.py --gpu $1 \
  --solver models/Pascal/VGG16/az-net/solver.prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --imdb voc_$2 \
  --cfg experiments/cfgs/vgg16_voc$2_skip.yml \
  --iters 160000

# frcnn: freeze conv layers
time ./tools/train_det_net.py --gpu $1 \
  --solver models/Pascal/VGG16_skip/frcnn/frozen/solver.prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --def models/Pascal/VGG16/az-net/test.prototxt \
  --def_fc models/Pascal/VGG16/az-net/test_fc.prototxt \
  --net output/az-net/voc_$2/vgg16_az_net_iter_160000.caffemodel \
  --imdb voc_$2 \
  --cfg experiments/cfgs/vgg16_voc$2_skip.yml \
  --iters 40000

# frcnn: finetune conv4-conv5
time ./tools/train_det_net.py --gpu $1 \
  --solver models/Pascal/VGG16_skip/frcnn/finetune/solver.prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --def models/Pascal/VGG16/az-net/test.prototxt \
  --def_fc models/Pascal/VGG16/az-net/test_fc.prototxt \
  --net output/az-net/voc_$2/vgg16_az_net_iter_160000.caffemodel \
  --imdb voc_$2 \
  --cfg experiments/cfgs/vgg16_voc$2_skip.yml \
  --iters 100000

# az-net: shared conv layers
time ./tools/train_az_net.py --gpu $1 \
  --solver models/Pascal/VGG16/az-net/shared/solver.prototxt \
  --weights output/az-net/voc_$2/vgg16_fast_rcnn_skip_finetune_iter_160000.caffemodel \
  --imdb voc_$2 \
  --cfg experiments/cfgs/vgg16_voc$2_skip.yml \
  --iters 160000

# frcnn: shared conv layers
time ./tools/train_det_net.py --gpu $1 \
  --solver models/Pascal/VGG16_skip/frcnn/shared/solver.prototxt \
  --weights output/az-net/voc_$2/vgg16_fast_rcnn_skip_finetune_iter_160000.caffemodel \
  --def models/Pascal/VGG16_skip/az-net/test.prototxt \
  --def_fc models/Pascal/VGG16_skip/az-net/test_fc.prototxt \
  --net output/az-net/voc_$2/vgg16_az_net_shared_iter_160000.caffemodel \
  --imdb voc_$2 \
  --cfg experiments/cfgs/vgg16_voc$2_skip.yml \
  --norm \
  --iters 80000

# testing with layer sharing
time ./tools/test_shared.py --gpu $1 \
  --def_fc_frcnn models/Pascal/VGG16_skip/frcnn/test_fc.prototxt \
  --net_frcnn output/az-net/voc_$2/vgg16_fast_rcnn_skip_shared_iter_80000.caffemodel \
  --def_az models/Pascal/VGG16_skip/az-net/test.prototxt \
  --def_fc_az models/Pascal/VGG16_skip/az-net/test_fc.prototxt \
  --net_az output/az-net/voc_$2/vgg16_az_net_shared_iter_160000.caffemodel \
  --cfg experiments/cfgs/vgg16_voc$3_skip.yml \
  --imdb voc_$3 \
  --comp
