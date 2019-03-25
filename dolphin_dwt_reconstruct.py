#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 12:36:48 2019

@author: dilber
"""

import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt

from dolphin_dwt_ae import autoencoder

filename = '/home/dilber/Desktop/DL/network_dataset/dolphins/dolphins_gt.gml'
G_dolphin = nx.read_gml(filename)
B_dolphin = nx.modularity_matrix(G_dolphin)
#------------------------------------------------------------------------------
encoder, ae = autoencoder(B_dolphin)

epochs = 100
train_batch_size = 62

history = ae.fit(B_dolphin, B_dolphin, batch_size=train_batch_size, epochs= epochs)

recons = encoder.predict(B_dolphin)

#------------------------------------------------------------------------------

B_dolphin_X = np.array(recons)
kmeans = KMeans(n_clusters=4, n_init=100, random_state=0)
kmeans.fit(B_dolphin_X)
X_ae = kmeans.labels_

#------------------------------------------------------------------------------

c_attributes = nx.get_node_attributes(G_dolphin, 'value')
c_groups = []

for i, val in enumerate(c_attributes.values()):
    c_groups.append(val)

X_gt = np.array(c_groups)
#------------------------------------------------------------------------------
print(history.history.keys())
plt.plot(history.history['loss'])
plt.title('Dolphin')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

#------------------------------------------------------------------------------
metrics.normalized_mutual_info_score(X_gt, X_ae, average_method='arithmetic')













