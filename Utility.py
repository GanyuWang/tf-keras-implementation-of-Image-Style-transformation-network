# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 16:57:03 2020

@author:Ganyu Wang

Utility

"""

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf

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



