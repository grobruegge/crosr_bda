# Classification Reconstruction Open-Set Recognition: Python Implementation

This repository aims to provide a lightweighted pytorch implementation of the Classification Reconstruction Open-Set Recognition (CROSR) framework introduced by the following paper: https://arxiv.org/pdf/1812.04246.pdf

## Installation

### Packages

To get started, the following packages are required:

- PyTorch: https://pytorch.org/get-started/locally/
- Numpy
- SciPy
- SKlearn

You can install them using either pip or conda or whatever. It it recommended to use a virtual environment.

Additionally, this repository requires LibMR for some stuff. This libary unfortunally does not come out-of-the-box, but you can install with the following steps:

1. Install Cython 
2. Clone the following repository into this projects folder: https://github.com/Vastlab/libMR/tree/master
3. The README of this repository includes a instruction how to install LibMR

FYI: There also seems to be a option to install LibMR using pip (https://pypi.org/project/libmr/), but this did not work for me.

### Datasets

The training of the neural network and the evaluation runs out-of-the-box with CIFAR-10, but it is possible to change the dataset (e.g., MNIST, TinyImageNet, ...). Simply change the torchvision.dataset to the desired one (If it is not supported by torchvision, you need to get creative).

The outlier datasets are downloaded from the following repository: https://github.com/facebookresearch/odin. Without changing anything, the code runs with TinyImageNet (Crop) as outliers (https://www.dropbox.com/s/avgm2u562itwpkl/Imagenet.tar.gz) but the other ones can be downloaded and used as well (just change the hard-coded path in the code).

1. Use wget to download the dataset into the ./data folder: wget https://www.dropbox.com/s/avgm2u562itwpkl/Imagenet.tar.gz 
2. Unpack: tar xvfz Imagenet.tar.gz

## Usage

There are only two scripts so the usage is pretty straight forward:

`train_dhr_nn`: Training script for the Deep Hierarchical Network (DHR) defined in the paper

This repository also provides pre-trained weights: `dhr_net.pt`

`compute_openmax`: The Mean Actiation Vectors (MAV) and distances to those are calculated and used to fit the Meta-Recognition System (MRS) consisting of the Weibull distributions. This is then used to implement the OpenMax framework.