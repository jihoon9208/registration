
import numpy as np
from sklearn.neighbors import NearestNeighbors
import numpy.matlib

#------------------------------------------------------------------------------
def compute_graph_nn(xyz, k_nn):
    """compute the knn graph"""
    
    num_ver = xyz.shape[0]
    graph = dict([("is_nn", True)])
    nn = NearestNeighbors(n_neighbors=k_nn+1, algorithm='kd_tree').fit(xyz)
    distances, neighbors = nn.kneighbors(xyz)
    neighbors = neighbors[:, 1:]
    distances = distances[:, 1:]
    source = np.matlib.repmat(range(0, num_ver), k_nn, 1).flatten(order='F')
    #save the graph
    graph["source"] = source.flatten().astype('int64')
    graph["target"] = neighbors.flatten().astype('int64')
    graph["distances"] = distances.flatten().astype('float32')
    return graph