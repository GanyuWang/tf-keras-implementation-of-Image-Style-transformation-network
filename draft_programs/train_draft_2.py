# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 14:28:49 2020

@author: Ganyu Wang

Train draft 2.

use the dataset intel recognition. 

train a identify transformation model. 

Predict.

Visualization. 


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



train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        '../intel-image-classification/seg_train/seg_train',
        target_size=(256, 256),
        batch_size=1,
        class_mode= "input")


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

inputs = keras.Input(shape=(256, 256, 3))
outputs = Identical_transform_model(inputs)
model = keras.Model(inputs=inputs, outputs=outputs)



#%% define loss function. 

def simple_loss_fn(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`


#%% compile. 
model.compile(optimizer='adam',
              loss=simple_loss_fn,
              metrics=['accuracy'])

#%% train. 

checkpoint_filepath = 'checkpoint_draft/'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)

model.fit(train_generator, callbacks=[model_checkpoint_callback])

#%%
model.save("draft2_identical_transform_model")


#%% save and reconstruct the model. 
reconstruct_model = keras.models.load_model("draft2_identical_transform_model")


#%% predict 
import PIL
import numpy as np

def load_image(path):
    """
    load image from path, to a 
    """
    t = PIL.Image.open(path)
    t = t.resize((256, 256))
    t = np.asarray(t)
    t = t/256
    t = t.reshape(1, 256, 256, 3)
    return t
    
test_img = load_image("content_img_test/3.jpg")
pred_img = reconstruct_model.predict(test_img)


#%% visualization. 
import matplotlib.pyplot as plt

plt.figure()
plt.imshow(test_img[0,])

plt.figure()
plt.imshow(pred_img[0,])


