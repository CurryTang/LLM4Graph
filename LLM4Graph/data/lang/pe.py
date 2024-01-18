import torch_geometric
import torch
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import eigs, eigsh
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from LLM4Graph.utils.misc import pickle_load, pickle_save
import os.path as osp


def random_walk(A, n_iter):
    # Geometric diffusion features with Random Walk
    Dinv = A.sum(dim=-1).clamp(min=1).pow(-1).unsqueeze(-1)  # D^-1
    RW = A * Dinv
    M = RW
    M_power = M
    # Iterate
    PE = [torch.diagonal(M)]
    for _ in range(n_iter-1):
        M_power = torch.matmul(M_power, M)
        PE.append(torch.diagonal(M_power))
    PE = torch.stack(PE, dim=-1)
    return PE


def RWSE(edge_index, pos_enc_dim, num_nodes):
    """
        Initializing positional encoding with RWSE
    """
    if edge_index.size(-1) == 0:
        PE = torch.zeros(num_nodes, pos_enc_dim)
    else:
        A = torch_geometric.utils.to_dense_adj(
            edge_index, max_num_nodes=num_nodes)[0]
        PE = random_walk(A, pos_enc_dim)
    return PE


def LapPE(edge_index, pos_enc_dim, num_nodes):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    degree = torch_geometric.utils.degree(edge_index[0], num_nodes)
    A = torch_geometric.utils.to_scipy_sparse_matrix(
        edge_index, num_nodes=num_nodes)
    N = sp.diags(np.array(degree.clip(1) ** -0.5, dtype=float))
    L = sp.eye(num_nodes) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    PE = torch.from_numpy(EigVec[:, 1:pos_enc_dim+1]).float()
    if PE.size(1) < pos_enc_dim:
        zeros = torch.zeros(num_nodes, pos_enc_dim)
        zeros[:, :PE.size(1)] = PE
        PE = zeros
    return PE


def cal_lap_pe(k, data, is_undirected=True, cache=True, cfg = None, cache_path="/tmp"):
    """
        Calculate the Laplacian positional encoding, adapted from Pyg
    """
    if cache and cfg != None and osp.exists(osp.join(cache_path, "lap_pe_{}_{}.pkl".format(cfg.dataset.name, k))):
        pe = pickle_load(osp.join(cache_path, "lap_pe_{}_{}.pkl".format(cfg.dataset.name, k)))
        return pe
    assert data.edge_index is not None
    num_nodes = data.x.size(0)
    assert num_nodes is not None

    edge_index, edge_weight = get_laplacian(
        data.edge_index,
        data.edge_weight,
        normalization='sym',
        num_nodes=num_nodes,
    )

    L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)


    eig_fn = eigs if not is_undirected else eigsh

    eig_vals, eig_vecs = eig_fn(  # type: ignore
        L,
        k=k + 1,
        which='SR' if not is_undirected else 'SA',
        return_eigenvectors=True,
    )

    eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
    pe = torch.from_numpy(eig_vecs[:, 1:k + 1])
    sign = -1 + 2 * torch.randint(0, 2, (k, ))
    pe *= sign
    if cache and cfg != None:
        pickle_save(pe, osp.join(cache_path, "lap_pe_{}_{}.pkl".format(cfg.dataset.name, k)))
    return pe 
