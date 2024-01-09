from torch_geometric.datasets import Planetoid, Actor
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Data
import requests
import os.path as osp
import numpy as np
import torch


fixed_urls = {
    'squirrel': 'https://github.com/yandex-research/heterophilous-graphs/raw/main/data/squirrel_filtered.npz',
    'chameleon': 'https://github.com/yandex-research/heterophilous-graphs/raw/main/data/chameleon_filtered.npz'
}


def get_data(cfg):
    """
        For every dataset, we change the train_mask, test_mask, val_mask to a list of masks train_masks, test_masks, val_masks
    """
    if cfg.dataset.name == 'chameleon' or cfg.dataset.name == 'squirrel':
        # wiki = WikipediaNetwork(root=cfg.dataset.root, name=cfg.dataset.name)
        if not osp.exists(cfg.dataset.root, cfg.dataset.name):
            print('Downloading {} dataset...'.format(cfg.dataset.name))
            r = requests.get(fixed_urls[cfg.dataset.name])
            with open(osp.join(cfg.dataset.root, cfg.dataset.name + '_filtered.npz'), 'wb') as f:
                f.write(r.content)
        data = np.load(osp.join(cfg.dataset.root, cfg.dataset.name + '_filtered.npz'))
        features = torch.from_numpy(data['node_features'])
        labels = torch.from_numpy(data['node_labels'])
        edge_index = torch.from_numpy(data['edges'].T)
        train_masks = torch.from_numpy(data['train_masks'])
        test_masks = torch.from_numpy(data['test_masks'])
        val_masks = torch.from_numpy(data['val_masks'])
        data = Data(x=features, y=labels, edge_index=edge_index, train_masks=train_masks, test_masks=test_masks, val_masks=val_masks)
        return data
    elif cfg.dataset.name == 'cora' or cfg.dataset.name == 'citeseer' or cfg.dataset.name == 'pubmed':
        dataset = Planetoid(root=cfg.dataset.root, name=cfg.dataset.name)
        data = dataset[0]
        data.train_masks = [data.train_mask]
        data.test_masks = [data.test_mask]
        data.val_masks = [data.val_mask]
        del data.train_mask, data.test_mask, data.val_mask
        return data
    elif 'ogbg' in cfg.dataset.name:
        ## graph-level OGB datasets
        pass
    else:
        print('Dataset {} not supported'.format(cfg.dataset.name))
        raise NotImplementedError
