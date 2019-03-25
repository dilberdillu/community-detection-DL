#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 14:12:59 2018

@author: dillu
"""

import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
from football_dwt_ae import autoencoder

filename = '/home/dilber/Desktop/DL/network_dataset/football/football.gml'
G_football = nx.read_gml(filename)
B_football = nx.modularity_matrix(G_football)

#------------------------------------------------------------------------------
encoder, ae = autoencoder(B_football)

epochs = 100
train_batch_size = 64

history = ae.fit(B_football, B_football, batch_size=train_batch_size, epochs=epochs)

recons = encoder.predict(B_football)

#------------------------------------------------------------------------------
B_football_X = np.array(recons)
kmeans = KMeans(n_clusters=12, n_init=100, random_state=0)
kmeans.fit(B_football_X)
X_ae = kmeans.labels_
#---------------------------------------------------------|Ground Truth|-------
c_attributes = nx.get_node_attributes(G_football,'value')
c_groups = []

for i, val in enumerate(c_attributes.values()):
    c_groups.append(val)
        
X_gt = np.array(c_groups)

#------------------------------------------------------------------------------
plt.plot(history.history['loss'])
plt.title('Football')
plt.ylabel('loss')
plt.xlabel('epoch')
#---------------------------
metrics.normalized_mutual_info_score(X_gt, X_ae, average_method='arithmetic')


