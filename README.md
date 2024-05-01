# CAP6610 Machine Learning Project

CIFAR- 10 image dataset classification using Convolutional neural networks.

## Overview
In this project, non neural networks and neural networks algorithms are implemented to classify the multi-class images in CIFAR-10 dataset.

### Dataset

https://www.cs.toronto.edu/~kriz/cifar.html

### Models used

1. LightGBM
2. Random Forest
3. Logistic Regression
4. CNN
5. AlexNet
6. VGG19
7. ResNet50 

### Programming language

Python

### IDE

Google Colab

### Implementation - File details

To implement this project, We have created few Python file in Google Colab as shown below:

cifar_rf.ipynb: In this file, We performed data preprocessing and implemented Random Forest,and evaluated model.
cifar_LG.ipynb: In this file, We performed data preprocessing and implemented Logistic Regression, and evaluated model.
LGBM.ipynb: IN this file, we performed data preprocessing and implemented LightGBM, and evaluated the model.
CNN.ipynb: In this file, we performed data preprocessing and implemented CNN model, and evaluated model.
AlexNet.ipynb: In this file , we performed data preprocessing and implemented AlexNet model, and evaluated the model.
VGG19.ipynb: in this file, we performed data preprocessing and implemented VGG19 model, and evaluated the model.
RESNET50.ipynb: In this file, we performed data preprocessing and implemented RESNET50 model, and evaluated the model.

### Requirements

You can run the above files in google colab to obtain results. You need to have the following installations:

seaborn
NumPy
matplotlib.pyplot
TensorFlow for keras
tensorflow.keras.layers for Conv2D, Maxpooling2D models, layers, regularizes
keras.application for VGG19
keras.Dataset for cifar10
keras.model for sequential
keras.layers for dense, flatten, dropout, upsampling2D, BatchNormalization, Activation
keras.optimiser for Adam, SGD
keras.preprocessing.image for image to array, array to img, ImageDataGenerator
keras.applications.vgg19 for process_input
keras.util for to categorical
sklearn.metrics for confusion matrix, accuracy_score
sklearn.model.selection for train_test_split
keras.callbacks for earlystopping, reduceLRonPlateau

Note : A GPU is required for faster execution, as CPUs take more time to execute. If you're using Colab mode, please use V100 GPU for execution.

First, you need to run the each notebook file mentioned above where you can see the results and also graphs below.