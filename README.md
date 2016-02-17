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

## Installation

To install, use the following steps:

1. Install Caffe and all its dependencies. 1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  ```

2. Clone the AZ-Net repository. Make sure to use the `--recursive` flag

  ```Shell
  # Make sure to clone with --recursive
  git clone --recursive https://github.com/rbgirshick/fast-rcnn.git
  ```

3. Build the Cython modules

    ```Shell
    cd $ROOT/lib
    make
    ```

4. Build Caffe and pycaffe
    ```Shell
    cd $ROOT/caffe-fast-rcnn
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j8 && make pycaffe
    ```

5. Fetch ImageNet models

    ```Shell
    cd $ROOT
    ./data/scripts/fetch_imagenet_models.sh
    ```
    
    See `data/README.md` for details.

6. To train and test models, use scripts in $ROOT/experiments/scripts
