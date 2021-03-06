# Deep Image Matting
Image Matting. Given an image, the code in this project can separate its foreground and background components.
This repository is to reproduce [Deep image matting](https://arxiv.org/abs/1703.03872) and is a modification to the codes used by foamliu.

## History

Hi all,
	All of us must have tried finding available github codes to train a Deep Image Matting model. The most common repository we come across is [foamliu's repo](https://github.com/foamliu/Deep-Image-Matting). The repository is no longer being maintained and the documentation doesn't seem to be sufficient to help with the training. In light of this, I have created this repository for Deep Image matting training in tensorflow which I shall maintain and help solve your problems. I have also added various functionalities in the train.py to help configure your training.
I have made the training process very simple. All you need to do is run 'python train.py' and you are all set. Appropriate comments have been added to ensure you can tune the parameters according to your use. If you get stuck anywhere, feel free to reach out.

## Dependencies
- [NumPy 1.18.5](https://pypi.org/project/numpy/1.18.5/)
- [Tensorflow 2.2.0](https://www.tensorflow.org/)
- [Keras 2.4.3](https://keras.io/#installation)
- [OpenCV 4](https://opencv.org/releases/)

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

### ImageNet Pretrained Models (Must-do)
Download [VGG16](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5) into "models" folder.

### Results using the pre-trained model:
Download foamliu's pre-trained Deep Image Matting [Model](https://github.com/foamliu/Deep-Image-Matting/releases/download/v1.0/final.42-0.0398.hdf5).

Image/Trimap | Output/GT | New BG/Compose | 
|---|---|---|
|![image](https://github.com/adumrewal/imageMatting/raw/master/images/0_image.png)  | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/0_out.png)   | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/0_new_bg.png) |
|![image](https://github.com/adumrewal/imageMatting/raw/master/images/0_trimap.png) | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/0_alpha.png) | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/0_compose.png)|
|![image](https://github.com/adumrewal/imageMatting/raw/master/images/1_image.png)  | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/1_out.png)   | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/1_new_bg.png) | 
|![image](https://github.com/adumrewal/imageMatting/raw/master/images/1_trimap.png) | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/1_alpha.png) | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/1_compose.png)|
|![image](https://github.com/adumrewal/imageMatting/raw/master/images/2_image.png)  | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/2_out.png)   | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/2_new_bg.png) |
|![image](https://github.com/adumrewal/imageMatting/raw/master/images/2_trimap.png) | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/2_alpha.png) | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/2_compose.png)|
|![image](https://github.com/adumrewal/imageMatting/raw/master/images/3_image.png)  | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/3_out.png)   | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/3_new_bg.png) |
|![image](https://github.com/adumrewal/imageMatting/raw/master/images/3_trimap.png) | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/3_alpha.png) | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/3_compose.png)|
|![image](https://github.com/adumrewal/imageMatting/raw/master/images/4_image.png)  | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/4_out.png)   | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/4_new_bg.png) |
|![image](https://github.com/adumrewal/imageMatting/raw/master/images/4_trimap.png) | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/4_alpha.png) | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/4_compose.png)|
|![image](https://github.com/adumrewal/imageMatting/raw/master/images/5_image.png)  | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/5_out.png)   | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/5_new_bg.png) |
|![image](https://github.com/adumrewal/imageMatting/raw/master/images/5_trimap.png) | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/5_alpha.png) | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/5_compose.png)|
|![image](https://github.com/adumrewal/imageMatting/raw/master/images/6_image.png)  | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/6_out.png)   | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/6_new_bg.png) |
|![image](https://github.com/adumrewal/imageMatting/raw/master/images/6_trimap.png) | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/6_alpha.png) | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/6_compose.png)|
|![image](https://github.com/adumrewal/imageMatting/raw/master/images/7_image.png)  | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/7_out.png)   | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/7_new_bg.png) |
|![image](https://github.com/adumrewal/imageMatting/raw/master/images/7_trimap.png) | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/7_alpha.png) | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/7_compose.png)|
|![image](https://github.com/adumrewal/imageMatting/raw/master/images/8_image.png)  | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/8_out.png)   | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/8_new_bg.png) |
|![image](https://github.com/adumrewal/imageMatting/raw/master/images/8_trimap.png) | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/8_alpha.png) | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/8_compose.png)|
|![image](https://github.com/adumrewal/imageMatting/raw/master/images/9_image.png)  | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/9_out.png)   | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/9_new_bg.png) |
|![image](https://github.com/adumrewal/imageMatting/raw/master/images/9_trimap.png) | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/9_alpha.png) | ![image](https://github.com/adumrewal/imageMatting/raw/master/images/9_compose.png)|

