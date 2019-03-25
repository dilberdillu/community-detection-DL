#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 04:28:53 2018

@author: dillu
"""

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

from polblogs_dwt_ae import autoencoder

filename = '/home/dilber/Desktop/DL/network_dataset/polblogs/polblogs.gml'
G_polblogs = nx.read_gml(filename)
G_polblogs = G_polblogs.to_undirected()
G_polblogs = nx.Graph(G_polblogs)
B_polblogs = nx.modularity_matrix(G_polblogs)


#------------------------------------------------------------------------------
encoder, ae = autoencoder(B_polblogs)

epochs = 100
train_batch_size = 1490
history = ae.fit(B_polblogs, B_polblogs, batch_size=train_batch_size, epochs=epochs)
recons = encoder.predict(B_polblogs)
#--------------------------------------------------------------|Kmeans|--------
B_polblogs_X = np.array(recons)
kmeans = KMeans(n_clusters=2, n_init=200, random_state=0)
kmeans.fit(B_polblogs_X)
X_ae = kmeans.labels_
#------------------------------------------------------------------------------
#---------------------------------------------------------------|Ground Truth|-
c_attributes = nx.get_node_attributes(G_polblogs,'value')
c_groups = []

for i, val in enumerate(c_attributes.values()):
    c_groups.append(val)
        
X_gt = np.array(c_groups)
#------------------------------------------------------------------------------
print(history.history.keys())
plt.plot(history.history['loss'])
plt.title('Polblogs')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

#------------------------------------------------------------------------------
metrics.normalized_mutual_info_score(X_gt, X_ae, average_method='arithmetic')

