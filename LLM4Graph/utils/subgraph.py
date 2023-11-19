import torch
from torch_sparse import SparseTensor



def k_hop_subgraph(edge_index, num_nodes, num_hops, is_directed=False):
    r"""Generates k-hop subgraphs for all nodes in a graph.

    This function computes the k-hop subgraph for every node in a graph,
    represented by its edge indices. It supports both directed and undirected
    graphs. For directed graphs, edges are made bidirectional before computing
    the subgraphs.

    Args:
        edge_index (Tensor): The edge indices of the graph as a 2 x num_edges
            tensor, where each column represents an edge.
        num_nodes (int): The number of nodes in the graph.
        num_hops (int): The number of hops to consider for subgraph creation.
        is_directed (bool, optional): Flag indicating if the graph is directed.
            If `True`, edges are treated as bidirectional. (default: :obj:`False`)

    Returns:
        Tensor: A boolean tensor of shape N x N, where N is the number of nodes.
            Each entry [i, j] is `True` if node j is within `num_hops` of node i,
            and `False` otherwise.

    Example:
        >>> edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        >>> num_nodes = 3
        >>> num_hops = 2
        >>> k_hop_subgraph(edge_index, num_nodes, num_hops)
        tensor([[True, True, True],
                [True, True, True],
                [True, True, True]])
    """
    # Function implementation...
    # return k-hop subgraphs for all nodes in the graph
    if is_directed:
        row, col = edge_index
        birow, bicol = torch.cat([row, col]), torch.cat([col, row])
        edge_index = torch.stack([birow, bicol])
    else:
        row, col = edge_index
    sparse_adj = SparseTensor(
        row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    # each one contains <= i hop masks
    hop_masks = [torch.eye(num_nodes, dtype=torch.bool,
                           device=edge_index.device)]
    hop_indicator = row.new_full((num_nodes, num_nodes), -1)
    hop_indicator[hop_masks[0]] = 0
    for i in range(num_hops):
        next_mask = sparse_adj.matmul(hop_masks[i].float()) > 0
        hop_masks.append(next_mask)
        hop_indicator[(hop_indicator == -1) & next_mask] = i+1
    hop_indicator = hop_indicator.T  # N x N
    node_mask = (hop_indicator >= 0)  # N x N dense mask matrix
    return node_mask


