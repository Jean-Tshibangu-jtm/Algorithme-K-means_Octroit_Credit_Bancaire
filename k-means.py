# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 21:54:13 2019

@author: Formateur IT
"""

import numpy as np

import pandas as pd 

X = pd.read_csv('credit_bank.csv').values

from sklearn.cluster import KMeans


kmeans = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=3, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=None, tol=0.0001, verbose=0)


kmeans.fit(X)
print(kmeans.cluster_centers_)

y_kmeans = kmeans.fit_predict(X)


import matplotlib.pyplot as plt

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'blue', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'red', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'yellow', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'green', label = 'Centroids')
plt.title('Clusters de demandeur_credit')
plt.xlabel('epargne en millier')
plt.ylabel('score_bank')
plt.legend()
plt.show()


kmeans.inertia_


clusters = [1,2,3,4,5,6,7,8,9,10]

inertia_values = []

for cluster in clusters:
    
    kmeans = KMeans(n_clusters = cluster)
    
    kmeans.fit(X)
    
    inertia_values.append(kmeans.inertia_)
    
import seaborn as sns

sns.pointplot(x = clusters, y = inertia_values)
plt.xlabel('Nombre de Clusters')
plt.ylabel("valeur d'Inertie")
plt.title("Nombre de Clusters Vs. valeur d' Inertie")
plt.show()

















