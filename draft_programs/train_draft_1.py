# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 20:31:10 2020

@author: Ganyu Wang

Train Draft

use the simplest dataset, CIFAR 10.

train a identify transformation model. 

Predict.

Visualization. 

"""

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import keras

import keras
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.constraints import maxnorm

from keras.datasets import cifar10
import tensorflow as tf
from keras import layers


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 255


#%% build models

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

def Identical_transform_model(y):
    """
    transform_model, channel last, 
        input  (batch_size, width, height, channel) -> (BS, 32, 32, 3)
        output (batch_size, width, height, channel) -> (BS, 32, 32, 3)
    
    """
    d = Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu')(y)
    d = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(d)
    d = residual_block(d, 64)
    d = residual_block(d, 64)
    d = residual_block(d, 64)
    d = Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=2)(d)
    d = Conv2DTranspose(3, (3, 3), activation='relu', padding='same', strides=1)(d)
    return d

inputs = keras.Input(shape=(32, 32, 3))
outputs = Identical_transform_model(inputs)
model = keras.Model(inputs=inputs, outputs=outputs)

#%% define loss function. 

def simple_loss_fn(y_true, y_pred):
    print(y_true)
    print(y_pred)
    
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=[1, 2, 3])  # Note the `axis=-1`


#%% compile. 

model.compile(optimizer='adam',
              loss=simple_loss_fn,
              metrics=['accuracy'])


#%% train. 

model.fit(x_train, x_train, epochs=2)


#%% predict 

n = 9
x = x_test[n:n+1, ] / 255
y = model.predict(x)



#%% visualization. 
import matplotlib.pyplot as plt

plt.figure()
plt.imshow(x[0,])

plt.figure()
plt.imshow(y[0,])






