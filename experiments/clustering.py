from typing import Dict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def get_clusters(train_datasets, args):
    vector_data = []
    for (train_x, train_y) in train_datasets:
        if args.data == 'x':
            vector_data.append(train_x.flatten())
        elif args.data == 'y':
            vector_data.append(train_y.flatten())

    n = len(vector_data)
    stacked = np.row_stack(vector_data)

    pca = PCA(n_components=min(args.pca, n))
    stacked = pca.fit_transform(stacked)

    kmeans = KMeans(n_clusters=args.n_clusters, random_state=0, init='k-means++', n_init='auto')
    kmeans.fit(stacked)

    clusters = [[] for _ in range(args.n_clusters)]
    for i, lab in enumerate(kmeans.labels_):
        clusters[lab].append(i)
    return clusters

def get_knn():
    pass
