from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.loader import NeighborSampler 
import torch

def get_graph_saint_loader(
    data, cfg
):
    graph_saint_loader = GraphSAINTRandomWalkSampler(
        data, batch_size=cfg.train.batch_size, walk_length=cfg.train.saint.walk_length, num_steps=cfg.train.saint.num_steps,
        sample_coverage=cfg.train.saint.sample_coverage, save_dir=cfg.dataset.root
    )

    return graph_saint_loader

def get_plain_data_loader(x, cfg, train_mask, val_mask, test_mask):
    """
        For NAGPhormer
    """
    train_loader = torch.utils.data.DataLoader(
        x[train_mask], batch_size=cfg.train.batch_size, batch_size = cfg.train.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        x[val_mask], batch_size=cfg.train.batch_size, batch_size = cfg.train.batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        x[test_mask], batch_size=cfg.train.batch_size, batch_size = cfg.train.batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader