#!/bin/bash

# usage: diag_voc gpu-id config-file prefix model train-set test-set
# default train set: voc_2007_trainval
# default test set: voc_2007_test
# default model: vgg16_az_net_shared_iter_160000.caffemodel 

gpu_id=$1
cfg_file=$2

trainset="voc_2007_trainval"
testset="voc_2007_test"
prefix="default"
model="vgg16_az_net_shared_iter_160000.caffemodel"

if [ $# -eq 0 ]
then
  echo Usage: diag_voc gpu-id config-file prefix model train-set test-set
  exit 1
fi

if [ $# -eq 6 ]
then
  prefix=$3
  model=$4
  trainset=$5
  testset=$6
elif [ $# -eq 4 ]
then
  prefix=$3
  model=$4
elif [ $# -eq 3 ]
then
  prefix=$3
fi

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/voc_diag.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/diagnose_prop.py --gpu $gpu_id \
  --def models/Pascal/VGG16/az-net/test.prototxt \
  --def_fc models/Pascal/VGG16/az-net/test_fc.prototxt \
  --net output/$prefix/$trainset/$model \
  --imdb $testset \
  --cfg experiments/cfgs/$cfg_file \
  --thresh output/$prefix/$trainset/$(basename $model .caffemodel)/thresh.pkl \
  --exp $prefix
  
