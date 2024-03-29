import torch
from torch_sparse import spspmm, spmm
from scipy.spatial.distance import cdist
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import scatter


def normalize_adj(edge_index, num_nodes, edge_weight = None):
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32,
                                 device=edge_index.device)
    
    row, col = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce='sum')
    deg_inv = 1.0 / deg
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    edge_weight = deg_inv[row] * edge_weight
    return edge_index, edge_weight


def _feature_similarity(features):
    return torch.from_numpy(1. - cdist(features, features, 'cosine'))


@torch.no_grad()
def get_feature_similarity(edge_index, features, k = 0):
    num_nodes = edge_index.shape[1]
    value = torch.ones(num_nodes)
    edge_index, value = normalize_adj(edge_index, num_nodes)
    if k == 0:
        ## directly use the feature matrix to compute the similarity
        return _feature_similarity(features.cpu())
    else:
        for _ in range(k - 1):
            edge_index, value = spspmm(edge_index, value, edge_index, value, num_nodes, num_nodes, num_nodes)
        aggr_features = spmm(edge_index, value, num_nodes, num_nodes, features)
        return _feature_similarity(aggr_features.cpu())

@torch.no_grad()
def get_propagated_features(edge_index, features, edge_attr = None, k = 0, normalize = True):
    """
        Input: edge_index, features
        Output: a list of propagated features
        k = 0 <-> center node
        k = 1 <-> 1-hop neighbors
    """
    # import ipdb; ipdb.set_trace()
    num_nodes = features.shape[0]
    num_edges = edge_index.shape[1]
    if edge_attr is None:
        value = torch.ones(num_edges)
    else:
        value = edge_attr
    if normalize:
        edge_index, value = normalize_adj(edge_index, num_nodes)
    results = []
    if k == 0:
        # features = features.reshape(num_nodes, 1, -1)
        results.append(features)
        return results
    else:
        results.append(features)
        aggr_features = features
        # import ipdb; ipdb.set_trace()
        for _ in range(k):
            # edge_index, value = spspmm(edge_index, value, edge_index, value, num_nodes, num_nodes, num_nodes)
            aggr_features = spmm(edge_index, value, num_nodes, num_nodes, aggr_features)
            results.append(aggr_features)
        return results