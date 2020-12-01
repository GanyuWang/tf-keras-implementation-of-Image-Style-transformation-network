# -*- coding: utf-8 -*-
"""
Created on Nov 27

@author: Ganyu Wang

Train Draft 4

use the dataset intel recognition. 

train a identify transformation model. 

save model, reconstructruct model.

Write the custom loss function again. 
    the custom loss function should return the 

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


assert tf.test.is_gpu_available()
assert tf.test.is_built_with_cuda()


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        '../intel-image-classification/seg_train/seg_train',
        target_size=(256, 256),
        batch_size=4,
        class_mode= "input",
        shuffle=False)


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
    d = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=2)(d)
    d = Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=2)(d)
    d = Conv2DTranspose(3, (9, 9), activation='relu', padding='same', strides=1)(d)
    
    return d

inputs = keras.Input(shape=(256, 256, 3))
outputs = Identical_transform_model(inputs)
transform_model = keras.Model(inputs=inputs, outputs=outputs)


#%% define loss function. 

def load_image(path, return_type="numpy"):
    """
    load image from path
        resize to (1, 256, 256, 3)
        
    """
    t = PIL.Image.open(path)
    t = t.resize((256, 256))
    t = np.asarray(t)
    t = t/256
    t = t.reshape(1, 256, 256, 3)
    
    if return_type == "tensor":
        return tf.constant(t, dtype="float32")
    return t

# style image.  
style_image = load_image("style_img/impression_sunrise.jpg", return_type='tensor')

# content layers
content_layers = ['block5_conv2'] 
num_content_layers = len(content_layers)

# style layers
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']
num_style_layers = len(style_layers)

# weight for style and content
style_weight=1e-2
content_weight=1e4


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

#
# transform_model = model.load_weight("model/transform_model_draft4.h5")

checkpoint_filepath = 'checkpoint/'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True)

transform_model.fit(train_generator, epochs=10, callbacks=[model_checkpoint_callback])

#%% save model
transform_model.save("model/transform_model_draft4.h5")



