# -*- coding: utf-8 -*-
"""
Created on Dec 7

@author: Ganyu Wang

Training the style transformation network. 

The experiment on epochs and 


"""

import numpy as np
import os
import PIL
import PIL.Image
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
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

style_image_path = "style_img/impression_sunrise.jpg"

# training parameters
batch_size = 2
epochs = 1

# whether load model weight from file
load_model = False
saved_model_path = "model/model_v9_epoch_3.h5"

# loss
style_weight = (1e-2) /2
content_weight = 1e3

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

num_content_layers = 1
# style loss.  
style_image = load_image(style_image_path, return_type='tensor')

num_style_layers = 5

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


#%% long train. 

if load_model == True:
    transform_model.load_weights(saved_model_path)

# check range.
for i in range(1,6):
    transform_model.fit(train_generator, epochs=epochs)
    transform_model.save("model/model_v25_epoch_%d.h5" % i) 

    
    test_image_path = "test_content_image/20123.jpg"
    
    test_img = load_image(test_image_path)
    pred_img = transform_model.predict(test_img)
   
    plt.figure()
    plt.imshow(pred_img[0,])
    
    # save predict figure
    plt.savefig("result/predict_v25_epoch_%d.png" % i)


