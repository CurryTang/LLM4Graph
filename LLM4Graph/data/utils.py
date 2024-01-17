from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.loader import NeighborSampler 
from LLM4Graph.utils.data_utils import seed_everything
from torch.utils.data import Dataset
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



class XYDataset(Dataset):
    def __init__(self, features, labels):
        """
        Initialization method for the dataset.
        
        :param features: A list or array of features.
        :param labels: A list or array of labels corresponding to the features.
        """
        self.features = features
        self.labels = labels

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.features)

    def __getitem__(self, index):
        """
        Generate one sample of data.
        
        :param index: The index of the item in the dataset.
        :return: A tuple containing the feature tensor and label tensor.
        """
        # Retrieve the feature and label at the specified index
        x = torch.tensor(self.features[index], dtype=torch.float32)
        y = torch.tensor(self.labels[index], dtype=torch.float32)

        return x, y


def get_plain_data_loader(data, cfg, train_mask, val_mask, test_mask):
    """
        For NAGPhormer
    """
    train_data = XYDataset(data.x[train_mask], data.y[train_mask])
    val_data = XYDataset(data.x[val_mask], data.y[val_mask])
    test_data = XYDataset(data.x[test_mask], data.y[test_mask])
    # loader_train_transform, loader_test_transform = get_loader_transforms_by_config(cfg)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=cfg.train.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=cfg.train.batch_size,  shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=cfg.train.batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader

def get_loader_from_config(data, cfg, i):
    if cfg.dataset.loader == 'torch':
        train_loader, val_loader, test_loader = get_plain_data_loader(
            data, cfg, data.train_masks[i], data.val_masks[i], data.test_masks[i]
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


def get_random_split_masks(num_nodes, train_ratio, val_ratio, seeds):
    train_masks = []
    val_masks = []
    test_masks = []
    for s in seeds:
        seed_everything(s)
        train_mask, val_mask, test_mask = generate_data_masks_torch(train_ratio, val_ratio, num_nodes)
        train_masks.append(train_mask)
        val_masks.append(val_mask)
        test_masks.append(test_mask)
    return train_masks, val_masks, test_masks