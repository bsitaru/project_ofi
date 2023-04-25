import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

def transform_data(train_datasets, args):
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
    return stacked
def get_clusters(train_datasets, args):
    stacked = transform_data(train_datasets, args)

    kmeans = KMeans(n_clusters=args.n_clusters, random_state=0, init='k-means++', n_init='auto')
    kmeans.fit(stacked)

    clusters = [[] for _ in range(args.n_clusters)]
    for i, lab in enumerate(kmeans.labels_):
        clusters[lab].append(i)
    return clusters

def get_kneighbours(train_datasets, args):
    stacked = transform_data(train_datasets, args)

    n = np.shape(stacked)[0]
    knn = NearestNeighbors(n_neighbors=min(args.neigh_size, n))
    knn.fit(stacked)

    neighs = knn.kneighbors(stacked, n_neighbors=min(args.neigh_size, n), return_distance=False).tolist()
    return neighs
