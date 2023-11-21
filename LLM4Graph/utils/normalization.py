import torch
from torch_geometric.utils import remove_self_loops, scatter



def normalize_adj(edge_index, num_nodes, edge_weight = None):
    """
    Function to normalize adjacency matrix for a graph.

    This function removes self loops in the graph, calculates the degree of each node,
    and normalizes the edge weights using the inverse of the node degree. If no edge weights
    are provided, they are assumed to be 1.

    Parameters:
    edge_index (Tensor): The edge indices.
    num_nodes (int): The number of nodes in the graph.
    edge_weight (Tensor, optional): The edge weights. Defaults to None, in which case all edge weights are assumed to be 1.

    Returns:
    edge_index (Tensor): The edge indices, with self loops removed.
    edge_weight (Tensor): The normalized edge weights.
    """
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