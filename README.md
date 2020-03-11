# Deep Image Matting
This repository is to reproduce [Deep image matting](https://arxiv.org/abs/1703.03872) and is a modification to the codes used by foamliu.

## History

Hi all,
	All of us must have tried finding available github codes to train a Deep Image Matting model. The most common repository we come across is [foamliu's repo](https://github.com/foamliu/Deep-Image-Matting). The repository is no longer being maintained and the documentation doesn't seem to be sufficient to help with the training. In light of this, I have created this repository for Deep Image matting training in tensorflow which I shall maintain and help solve your problems. I have also added various functionalities in the train.py to help configure your training.
I have made the training process very simple. All you need to do is run 'python train.py' and you are all set.

## Dependencies
- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow 1.9.0](https://www.tensorflow.org/)
- [Keras 2.1.6](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## Dataset

### Custom Dataset
- Use the 'data' folder to provide the dataset
- data/train_names.txt - Provide the names of the training files here. It will automatically detect the mask file names using the naming convention followed. You can refer to the existing sample data in the folders to know more.
- data/valid_names.txt - Provide the names of the validation files here.
- data/input - Contains the input files with merged foreground and background.
- data/mask - Contains the binary mask for foregrounds.

### Adobe Deep Image Matting Dataset
Follow the [instruction](https://sites.google.com/view/deepimagematting) to contact author for the dataset.

### MSCOCO
Go to [MSCOCO](http://cocodataset.org/#download) to download:
* [2014 Train images](http://images.cocodataset.org/zips/train2014.zip)


### PASCAL VOC
Go to [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) to download:
* VOC challenge 2008 [training/validation data](http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar)
* The test data for the [VOC2008 challenge](http://host.robots.ox.ac.uk/pascal/VOC/voc2008/index.html#testdata)

## ImageNet Pretrained Models (Must-do)
Download [VGG16](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5) into "models" folder.


