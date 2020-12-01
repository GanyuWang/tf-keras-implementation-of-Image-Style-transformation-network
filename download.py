# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 00:11:38 2020

@author:  Ganyu Wang

This is used for download dataset.

"""

import kaggle

kaggle.api.authenticate()

kaggle.api.dataset_download_files('puneet6060/intel-image-classification',
                                  path='../intel-image-classification', 
                                  unzip=True)


