#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 12:36:30 2019

@author: dilber
"""

import numpy as np
#import scipy.sparse as sp
#import tensorflow as tf
from keras.layers import Input, Dense, Dropout, Lambda
from keras.models import Model
from keras import optimizers
from keras import backend as K

from dwt_customLayer import DenseWeightTied
#------------------------------------------------------------------------------

def ce(y_true, y_pred):
    """ Sigmoid cross-entropy loss """
    return K.mean(K.binary_crossentropy(target=y_true,
                                        output=y_pred,
                                        from_logits=True),
                                        axis=-1)

def mvn(tensor):
    """Per row mean-variance normalization."""
    epsilon = 1e-6
    mean = K.mean(tensor, axis=1, keepdims=True)
    std = K.std(tensor, axis=1, keepdims=True)
    mvn = (tensor - mean) / (std + epsilon)
    return mvn

def autoencoder(adj, weight = None):
    h, w = adj.shape
    
    kwargs = dict(
            use_bias=True,
            kernel_initializer = 'glorot_normal',
            kernel_regularizer = None,
            bias_initializer='zeros',
            bias_regularizer=None,
            trainable=True)
    
    data = Input(shape=(w,), dtype=np.float32, name='data')
    noisy_data = Dropout(rate=0.2, name='drop0')(data)
    
    #first set of encoding
    encoded = Dense(32, activation='relu', name='encoded1', **kwargs)(noisy_data)
    encoded = Lambda(mvn, name='mvn1')(encoded)
    encoded = Dropout(rate=0.2, name='drop1')(encoded)
    
    #second encoding 
    encoded = Dense(16, activation='relu', name='encoded2', **kwargs)(encoded)
    encoded = Lambda(mvn, name='mvn2')(encoded)
    encoded = Dropout(rate=0.2, name = 'drop2')(encoded)
    
    encoder = Model([data], encoded)
    encoded1 = encoder.get_layer('encoded1')
    encoded2 = encoder.get_layer('encoded2')
    
    #first decoding
    decoded = DenseWeightTied(32, tie_to=encoded2, transpose=True, activation='relu',
                              name='decoded3')(encoded)
    decoded = Lambda(mvn, name='mvn3')(decoded)
    decoded = Dropout(rate=0.2, name='d-drop2')(decoded)
    
    #second decoding
    decoded = DenseWeightTied(w, tie_to=encoded1, transpose=True, activation='linear',
                              name='decoded')(decoded)
    decoded = Dropout(rate=0.2, name='d-drop1')(decoded)
    
    adam = optimizers.Adam(lr=0.001, decay=0.0)
    autoencoder = Model(inputs= [data], outputs= [decoded])
    autoencoder.compile(optimizer=adam, loss=ce, metrics=['accuracy'])
    
    return encoder, autoencoder


    
    
    
    
    
    