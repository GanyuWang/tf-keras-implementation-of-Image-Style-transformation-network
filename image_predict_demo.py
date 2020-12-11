# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 15:13:41 2020

@author: Ganyu Wang

Use the saved model to predict 

"""
import numpy as np
import os
import PIL
import PIL.Image

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from Model import Transform_model
from Utility import load_image
from Loss import extractor

import matplotlib.pyplot as plt

import time

assert tf.test.is_gpu_available()
assert tf.test.is_built_with_cuda()


#%% setting
model_path = "model/model_v0_1_1_epoch_5.h5"

test_image_path = "test_content_image/20123.jpg"
style_image_path = "style_img/impression_sunrise.jpg"

transfer_image_save_path = "test_content_image/predict_20123.jpg"


#%% build model
inputs = keras.Input(shape=(256, 256, 3))
outputs = Transform_model(inputs)
transform_model = keras.Model(inputs=inputs, outputs=outputs)

# load weights
transform_model.load_weights(model_path)


#%% image style transfer

test_img = load_image(test_image_path)
pred_img = transform_model.predict(test_img)
style_image = load_image(style_image_path)

plt.figure()
plt.imshow(style_image[0,])


plt.figure()
plt.imshow(test_img[0,])

plt.figure()
plt.imshow(pred_img[0,])

# save predict figure
plt.savefig("predict.png")


#%% time  calculation

time_list = []
for i in range(1000):
    t1 = time.time()
    pred_img = transform_model.predict(test_img)
    t2 = time.time()
    t = t2-t1
    time_list.append(t)

avg_time = np.average(time_list)
print("the average time is %f" % avg_time)



