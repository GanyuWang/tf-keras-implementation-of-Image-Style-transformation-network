# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 17:05:07 2020

@author: Ganyu Wang


define style and content loss extractor here 

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


#


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






class Style_Content_Loss():
    
    import tensorflow as tf

    from tensorflow import keras
    
    def __init__(self, style_image_path):
        # content layer name
        self.content_layers = ['block5_conv2'] 
        self.num_content_layers = len(self.content_layers)
        # style layer
        self.style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1', 
                        'block4_conv1', 
                        'block5_conv1']
        self.num_style_layers = len(self.style_layers)
        # style image
        self.style_image = load_image(style_image_path, return_type='tensor')
        
        # style_weight, content weight. 
        self.style_weight=1e-2
        self.content_weight=1e4
        
        self.extractor = StyleContentModel(style_layers, content_layers)

    # content input batch, the data from data generator, [batch, 256, 256, 3]
    def content_loss(self, content_input, transform_model_output):
        content_input_target = self.extractor(content_input)['content']  # a dict, content
        content_trans_target = self.extractor(transform_model_output)['content']
        content_loss = tf.add_n([tf.reduce_mean((content_trans_target[name]-content_input_target[name])**2, [1,2,3]) 
                                 for name in content_trans_target.keys()])
        content_loss /= self.num_content_layers     # divided by number of layers to standardlize the weight.
        return content_loss
    
    # style input is the image tf tensor (1, width, height, 3).
    def style_loss(self, style_input, transform_model_output):
        style_input_target = self.extractor(style_input)['style'] # a dict, {style_layer: tensor }
        style_trans_target = self.extractor(transform_model_output)['style']
        style_loss = tf.add_n([tf.reduce_mean((style_trans_target[name]-style_input_target[name])**2, [1,2]) 
                               for name in style_trans_target.keys()])
        style_loss /= self.num_style_layers
        return style_loss
    
    def style_content_loss(self, content_input, transform_model_output):
        total_loss = self.content_weight * self.content_loss(content_input, transform_model_output) \
                + self.style_weight * self.style_loss(self.style_image, transform_model_output) 
        return total_loss
        
    class StyleContentModel(tf.keras.models.Model):
        def __init__(self, style_layers, content_layers):
            super(StyleContentModel, self).__init__()
            self.vgg =  self.vgg_layers(style_layers + content_layers)
            self.style_layers = style_layers
            self.content_layers = content_layers
            self.num_style_layers = len(style_layers)
            self.vgg.trainable = False
            
        # VGG layers.
        def vgg_layers(self, layer_names):
            """ Creates a vgg model that returns a list of intermediate output values."""
            # Load our model. Load pretrained VGG, trained on imagenet data
            vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
            vgg.trainable = False
            outputs = [vgg.get_layer(name).output for name in layer_names]
            model = tf.keras.Model([vgg.input], outputs)
            return model
        
        def gram_matrix(self, input_tensor):
            result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
            input_shape = tf.shape(input_tensor)
            num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
            return result/(num_locations)
        
        def call(self, inputs):
            "Expects float input in [0,1]"
            inputs = inputs*255.0
            #preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs) # problem here
            preprocessed_input = inputs
            # eliminated preprocess
            
            outputs = self.vgg(preprocessed_input)
            style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                              outputs[self.num_style_layers:])
          
            style_outputs = [self.gram_matrix(style_output)
                             for style_output in style_outputs]
          
            content_dict = {content_name:value 
                            for content_name, value 
                            in zip(self.content_layers, content_outputs)}
          
            style_dict = {style_name:value
                          for style_name, value
                          in zip(self.style_layers, style_outputs)}
            
            return {'content':content_dict, 'style':style_dict}


















