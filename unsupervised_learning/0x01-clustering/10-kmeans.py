#!/usr/bin/env python3
"""
10. Hello, sklearn!
"""

import sklearn.cluster


def kmeans(X, k):
    """
    Args:
        X is a numpy.ndarray of shape (n, d) containing the dataset
        k is the number of clusters
    Returns: C, clss
        C is a numpy.ndarray of shape (k, d) containing the centroid
            means for each cluster
        clss is a numpy.ndarray of shape (n,) containing the index of
            the cluster in C that each data point belongs to
    """
    n, d = X.shape
    k_means = sklearn.cluster.KMeans(n_clusters=k)
    k_means.fit(X)
    clss = k_means.labels_
    C = k_means.cluster_centers_
    return C, clss
