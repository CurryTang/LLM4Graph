from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.loader import NeighborSampler 
from LLM4Graph.utils.data_utils import seed_everything
# from LLM4Graph.data.transforms import get_loader_transforms_by_config
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
    # loader_train_transform, loader_test_transform = get_loader_transforms_by_config(cfg)
    train_loader = torch.utils.data.DataLoader(
        x[train_mask], batch_size=cfg.train.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        x[val_mask], batch_size=cfg.train.batch_size,  shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        x[test_mask], batch_size=cfg.train.batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader

def get_loader_from_config(data, cfg, i):
    if cfg.dataset.loader == 'torch':
        train_loader, val_loader, test_loader = get_plain_data_loader(
            data.x, cfg, data.train_masks[i], data.val_masks[i], data.test_masks[i]
        )
    return train_loader, val_loader, test_loader


def generate_data_masks_torch(train_ratio, val_ratio, num_nodes):
    # Input validation
    if train_ratio + val_ratio > 1.0 or num_nodes <= 0:
        raise ValueError("Invalid input values.")

    # Creating a tensor of indices
    indices = torch.randperm(num_nodes)

    # Calculating the number of nodes in each set
    num_train = int(num_nodes * train_ratio)
    num_val = int(num_nodes * val_ratio)

    # Generating masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:num_train]] = True
    val_mask[indices[num_train:num_train + num_val]] = True
    test_mask[indices[num_train + num_val:]] = True

    return train_mask, val_mask, test_mask


def get_random_split_masks(num_nodes, train_ratio, val_ratio, num_seeds):
    train_masks = []
    val_masks = []
    test_masks = []
    for i in range(num_seeds):
        seed_everything(i)
        train_mask, val_mask, test_mask = generate_data_masks_torch(train_ratio, val_ratio, num_nodes)
        train_masks.append(train_mask)
        val_masks.append(val_mask)
        test_masks.append(test_mask)
    return train_masks, val_masks, test_masks