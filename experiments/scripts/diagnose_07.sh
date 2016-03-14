#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/voc_diag.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/diagnose_prop.py --gpu $1 \
  --def models/Pascal/VGG16/az-net/test.prototxt \
  --def_fc models/Pascal/VGG16/az-net/test_fc.prototxt \
  --net output/az-net/voc_2007_trainval/vgg16_az_net_iter_80000.caffemodel \
  --imdb voc_2007_test \
  --cfg experiments/cfgs/exp1.yml
