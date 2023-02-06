import numpy as np
from sklearn.cluster import MeanShift

def cluster_solutions(sol, *,
                      bandwidth=None, include_freq=False,
                      use_density_as_confidence=True):
    assert(isinstance(sol, np.ndarray))
    assert(sol.ndim == 2)
    ms = MeanShift(bandwidth=bandwidth)
    clustering = ms.fit(sol)
    cluster_centers = ms.cluster_centers_
    num_cluster = cluster_centers.shape[0]
    unique, counts = np.unique(clustering.labels_, return_counts=True)
    freq = np.zeros([num_cluster], dtype=float)
    if include_freq and use_density_as_confidence:
        for i in range(len(unique)):
            freq[unique[i]] = counts[i]
        freq = freq / sol.shape[0]
    else:
        freq = np.ones_like(freq)

    if include_freq:
        return cluster_centers, freq
    else:
        return cluster_centers
