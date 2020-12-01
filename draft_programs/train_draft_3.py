# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 16:16:38 2020

@author: Ganyu Wang

Train Draft 3

use the dataset intel recognition. 

train a identify transformation model. 

save model, reconstructruct model.


Predict.
Visualization. 


"""

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import keras

from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.constraints import maxnorm
from keras.datasets import cifar10
from keras import layers
from keras.preprocessing.image import ImageDataGenerator

from Model import Transform_model


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        'data/seg_train/seg_train',
        target_size=(256, 256),
        batch_size=32,
        class_mode= "input")


#%% build models

# Transfer model 
transform_model_func = Transform_model
inputs = keras.Input(shape=(256, 256, 3))
outputs = transform_model_func(inputs)
transform_model = keras.Model(inputs=inputs, outputs=outputs)


#%% define loss function. 

# get content loss and style loss from VGG net. 
content_layers = ['block5_conv2'] 
num_content_layers = len(content_layers)

# VGG layers.
def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

# input style image and resize.
def load_img_tf(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

# style loss.  
style_image = load_img_tf("data/style_img/impression_sunrise.jpg")
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']
num_style_layers = len(style_layers)

style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image*255)

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg =  vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False
    
    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs*255.0
        #preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs) # problem here
        preprocessed_input = inputs
        # eliminated preprocess
        
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                          outputs[self.num_style_layers:])
      
        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]
      
        content_dict = {content_name:value 
                        for content_name, value 
                        in zip(self.content_layers, content_outputs)}
      
        style_dict = {style_name:value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}
        
        return {'content':content_dict, 'style':style_dict}

extractor = StyleContentModel(style_layers, content_layers)

"""
The way to use extractor is 
input a image, 
the extractor will outputs the outputs, of that layer. 

Calling extractor will return embedding dictiona 
{'content': content_dict, 
 'style'  : style_dict}

content_dict = {content_name:value 
                        for content_name, value 
                        in zip(self.content_layers, content_outputs)}
      
style_dict = {style_name:value
              for style_name, value
              in zip(self.style_layers, style_outputs)}


the image input is a tf tensor, the shape is [1, width, height, 3]

style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

"""

style_weight=1e-2
content_weight=1e4

"""
    content_input: [Batch_size, 256, 256, 3]
"""

# content input batch, the data from data generator, [batch, 256, 256, 3]
def content_loss(content_input, transform_model_output):
    content_input_target = extractor(content_input)['content']  # a dict, content
    content_trans_target = extractor(transform_model_output)['content']
    content_loss = tf.add_n([tf.reduce_mean((content_trans_target[name]-content_input_target[name])**2) 
                             for name in content_trans_target.keys()])
    content_loss /= num_content_layers     # divided by number of layers to standardlize the weight.
    return content_loss.numpy()

# style input is the image tf tensor (1, width, height, 3).
def style_loss(style_input, transform_model_output):
    style_input_target = extractor(style_input)['style'] # a dict, {style_layer: tensor }
    style_trans_target = extractor(transform_model_output)['style']
    style_loss = tf.add_n([tf.reduce_mean((style_trans_target[name]-style_input_target[name])**2) 
                           for name in style_trans_target.keys()])
    style_loss /= num_style_layers
    return style_loss.numpy()
    
def style_content_loss(content_input, transform_model_output):
    
    total_loss = content_weight * content_loss(content_input, transform_model_output) \
            + style_weight * style_loss(style_image, transform_model_output) 
    return total_loss


#%% compile. 
transform_model.compile(
                loss=style_content_loss,
                optimizer='adam'
              )

#%% train. 
transform_model.fit(train_generator)

#%%
transform_model.save("transform_model")


#%% save and reconstruct the model. 
reconstruct_model = tf.keras.models.load_model("draft2_identical_transform_model", compile=False)


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
    
test_img = load_image("data/content_img/3.jpg")
pred_img = reconstruct_model.predict(test_img)


#%% visualization. 
import matplotlib.pyplot as plt

plt.figure()
plt.imshow(test_img[0,])

plt.figure()
plt.imshow(pred_img[0,])


