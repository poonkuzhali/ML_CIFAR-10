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

1. cifar_rf.ipynb: In this file, We performed data preprocessing and implemented Random Forest,and evaluated model.
2. cifar_LG.ipynb: In this file, We performed data preprocessing and implemented Logistic Regression, and evaluated model.
3. LGBM.ipynb: IN this file, we performed data preprocessing and implemented LightGBM, and evaluated the model.
4. CNN_1.ipynb: In this file, we performed data preprocessing and implemented CNN model, and evaluated model.
5. AlexNet.ipynb: In this file , we performed data preprocessing and implemented AlexNet model, and evaluated the model.
6. VGG19_data_aug.ipynb: in this file, we performed data preprocessing and implemented VGG19 model, and evaluated the model.
7. RESNET50.ipynb: In this file, we performed data preprocessing and implemented RESNET50 model, and evaluated the model.

### Requirements

You can run the above files in google colab to obtain results. You need to have the following installations:

1. seaborn
2. NumPy
3. matplotlib.pyplot
4. TensorFlow for keras
5. tensorflow.keras.layers for Conv2D, Maxpooling2D models, layers, regularizes
6. keras.application for VGG19
7. keras.Dataset for cifar10
8. keras.model for sequential
9. keras.layers for dense, flatten, dropout, upsampling2D, BatchNormalization, Activation
10. keras.optimiser for Adam, SGD
11. keras.preprocessing.image for image to array, array to img, ImageDataGenerator
12. keras.applications.vgg19 for process_input
13. keras.util for to categorical
14. sklearn.metrics for confusion matrix, accuracy_score
15. sklearn.model.selection for train_test_split
16. keras.callbacks for earlystopping, reduceLRonPlateau

Note : A GPU is required for faster execution, as CPUs take more time to execute. If you're using Colab mode, please use V100 GPU for execution.

First, you need to run the each notebook file mentioned above where you can see the results and also graphs below.
