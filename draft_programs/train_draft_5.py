# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 23:52:17 2020

@author: Ganyu Wang

Training a style transformation network. 


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

from Model import Transform_model
from Utility import load_image
from Loss import extractor

assert tf.test.is_gpu_available()
assert tf.test.is_built_with_cuda()

#%% Setting

# file path
dataset_path = '../intel-image-classification/seg_train/seg_train'

# training
batch_size = 1
epochs = 1

# loss
style_weight=1e-2
content_weight=1e4


#%% data generator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode= "input",
        shuffle=False)


#%% build model
inputs = keras.Input(shape=(256, 256, 3))
outputs = Transform_model(inputs)
transform_model = keras.Model(inputs=inputs, outputs=outputs)


#%% define loss function

# get content loss and style loss from VGG net. 
content_layers = ['block5_conv2'] 
num_content_layers = len(content_layers)
# style loss.  
style_image = load_image("style_img/impression_sunrise.jpg", return_type='tensor')
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']
num_style_layers = len(style_layers)

# content input batch, the data from data generator, [batch, 256, 256, 3]
def content_loss(content_input, transform_model_output):
    content_input_target = extractor(content_input)['content']  # a dict, content
    content_trans_target = extractor(transform_model_output)['content']
    content_loss = tf.add_n([tf.reduce_mean((content_trans_target[name]-content_input_target[name])**2, [1,2,3]) 
                             for name in content_trans_target.keys()])
    content_loss /= num_content_layers     # divided by number of layers to standardlize the weight.
    return content_loss

# style input is the image tf tensor (1, width, height, 3).
def style_loss(style_input, transform_model_output):
    style_input_target = extractor(style_input)['style'] # a dict, {style_layer: tensor }
    style_trans_target = extractor(transform_model_output)['style']
    style_loss = tf.add_n([tf.reduce_mean((style_trans_target[name]-style_input_target[name])**2, [1,2]) 
                           for name in style_trans_target.keys()])
    style_loss /= num_style_layers
    return style_loss

def style_content_loss(content_input, transform_model_output):
    total_loss = content_weight * content_loss(content_input, transform_model_output) \
            + style_weight * style_loss(style_image, transform_model_output) 
    return total_loss


#%% compile. 
transform_model.compile(
                optimizer='adam',
                loss=style_content_loss
              )


#%% train. 
checkpoint_filepath = 'checkpoint/'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)

transform_model.fit(train_generator, epochs=epochs, callbacks=[model_checkpoint_callback])


#%% save model
transform_model.save("model/model_draft_5")
transform_model.save("model/model_draft_5.h5")



