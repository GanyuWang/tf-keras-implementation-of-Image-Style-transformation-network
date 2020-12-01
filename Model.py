# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:01:31 2020

@author: Ganyu Wang
    
The function for building the transformation model. 

    
"""


import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def residual_block(y, nb_channels, _strides=(1, 1), _project_shortcut=False):
    shortcut = y

    # down-sampling is performed with a stride of 2
    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)

    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
    y = layers.BatchNormalization()(y)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = layers.Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    y = layers.add([shortcut, y])
    y = layers.LeakyReLU()(y)

    return y

def Transform_model(y):
    """
    transform_model, channel last, 
        input  (batch_size, width, height, channel) -> (BS, 256, 256, 3)
        output (batch_size, width, height, channel) -> (BS, 256, 256, 3)
    """
    
    d = Conv2D(32, (9, 9), input_shape=(256, 256, 3), padding='same', activation='relu')(y)
    d = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(d)
    d = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(d)
    d = residual_block(d, 128)
    d = residual_block(d, 128)
    d = residual_block(d, 128)
    d = residual_block(d, 128)
    d = residual_block(d, 128)
    d = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=2)(d)
    d = Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=2)(d)
    d = Conv2DTranspose(3, (9, 9), activation='relu', padding='same', strides=1)(d)
    
    return d



