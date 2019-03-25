#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 01:11:32 2018

@author: dillu
"""

import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
from polbooks_dwt_ae import autoencoder

filename = '/home/dilber/Desktop/DL/network_dataset/polbooks/polbooks.gml'
G_polbooks = nx.read_gml(filename)
B_polbooks = nx.modularity_matrix(G_polbooks)

#------------------------------------------------------------------------------
encoder, ae = autoencoder(B_polbooks)

epochs = 100
train_batch_size = 64

history = ae.fit(B_polbooks, B_polbooks, batch_size=train_batch_size, epochs=epochs)

recons = encoder.predict(B_polbooks)

#--------------------------------------------------------------|Kmeans|--------

B_polbooks_X = np.array(recons)
kmeans = KMeans(n_clusters=3, n_init=100, random_state=0)
kmeans.fit(B_polbooks_X)
X_ae = kmeans.labels_
#------------------------------------------------------------------------------
#---------------------------------------------------------------|GT|-----------
c_attributes = nx.get_node_attributes(G_polbooks,'value')
c_groups = []

for i, val in enumerate(c_attributes.values()):
    if val == 'l':
        c_groups.append(0)
    elif val == 'n':
        c_groups.append(1)
    else:
        c_groups.append(2)
        
X_gt = np.array(c_groups)
#------------------------------------------------------------------------------
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#----------------------------------------
metrics.normalized_mutual_info_score(X_gt, X_ae, average_method='arithmetic')

