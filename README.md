# AZ-Net

## Introduction
This github repository is an implementation of the AZ-Net detection method described in 
"Adaptive Object Detection Using Adjacency and Zoom Prediction" 

Created by Yongxi Lu at University of California, San Diego.

If you find this useful, please consider citing

  @article{lu2015adaptive,
      title={Adaptive Object Detection Using Adjacency and Zoom Prediction},
      author={Lu, Yongxi and Javidi, Tara and Lazebnik, Svetlana},
      journal={arXiv preprint arXiv:1512.07711},
      year={2015}
    }

## Instructions for installation and demos

To install, use the following steps:

1. Install all libraries necessary for Caffe. 

2. Download and compile the "Fast-RCNN" branch of Caffe. Code and detailed instructions available at https://github.com/rbgirshick/fast-rcnn

3. Fetch ImageNet model using the scripts in $ROOT$/data, following isntructions to create symlink to datasets.

4. Compile cython utilities by
cd $ROOT$/lib
make

5. To train and test models, use scripts in $ROOT$/experiments/scripts
