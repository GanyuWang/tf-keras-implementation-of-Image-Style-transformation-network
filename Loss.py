# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 17:05:07 2020

@author: Ganyu Wang


define style content loss extractor here

Extractor means input the image [batch_size, 256, 256, 3]
Output the activation value of the needed layers in VGG

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


#%% define the loss function. 


# style image.  
style_image = load_image("style_img/impression_sunrise.jpg", return_type='tensor')

# content layers (start with 1. )
content_layers = ['block5_conv2'] 
num_content_layers = len(content_layers)

# style layers
style_layers = ['block1_conv1', 
                'block2_conv1', 
                'block3_conv1', 
                'block4_conv1',
                'block5_conv1'
                ]

num_style_layers = len(style_layers)



# VGG layers.
def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

#
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


"""
    content_input: [Batch_size, 256, 256, 3]
"""

