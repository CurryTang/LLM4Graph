from torch_geometric.datasets import Planetoid, Actor, LRGBDataset, TUDataset
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Data
import requests
import os.path as osp
import os
import numpy as np
import torch
import scipy
from google_drive_downloader import GoogleDriveDownloader as gdd
from LLM4Graph.data.utils import generate_data_masks_torch, get_random_split_masks
from LLM4Graph.data.transforms import get_transforms_by_config, get_pre_transforms_by_config


fixed_urls = {
    'squirrel': 'https://github.com/yandex-research/heterophilous-graphs/raw/main/data/squirrel_filtered.npz',
    'chameleon': 'https://github.com/yandex-research/heterophilous-graphs/raw/main/data/chameleon_filtered.npz',
    'pokec': '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y'
}




def load_pokec_mat(data_dir, train_prop=0.5, val_prop=0.25):
    """ requires pokec.mat """
    if not osp.exists(f'{data_dir}/pokec/pokec.mat'):
        gdd.download_file_from_google_drive(
            file_id=fixed_urls['pokec'], \
            dest_path=f'{data_dir}/pokec/pokec.mat', showsize=True)

    try:
        fulldata = scipy.io.loadmat(f'{data_dir}/pokec/pokec.mat')
        edge_index = fulldata['edge_index']
        node_feat = fulldata['node_feat']
        label = fulldata['label']
    except:
        edge_index = np.load(f'{data_dir}/pokec/edge_index.npy')
        node_feat = np.load(f'{data_dir}/pokec/node_feat.npy')
        label = np.load(f'{data_dir}/pokec/label.npy')

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    node_feat = torch.tensor(node_feat).float()
    num_nodes = int(node_feat.shape[0])
    label = torch.tensor(label).flatten()

    dir = f'{data_dir}/pokec/split_0.5_0.25'
    if osp.exists(dir):
        train_mask = torch.as_tensor(np.loadtxt(dir + '/pokec_train.txt'), dtype=torch.long)
        valid_mask = torch.as_tensor(np.loadtxt(dir + '/pokec_valid.txt'), dtype=torch.long)
        test_mask = torch.as_tensor(np.loadtxt(dir + '/pokec_test.txt'), dtype=torch.long)
    else:
        os.makedirs(dir)
        train_mask, valid_mask, test_mask = generate_data_masks_torch(train_prop, val_prop, num_nodes)
        np.savetxt(dir + '/pokec_train.txt', train_mask, fmt='%d')
        np.savetxt(dir + '/pokec_valid.txt', valid_mask, fmt='%d')
        np.savetxt(dir + '/pokec_test.txt', test_mask, fmt='%d')
    
    data = Data(x=node_feat, edge_index=edge_index, y=label, train_mask=train_mask, val_mask=valid_mask, test_mask=test_mask)
    return data







def get_data(cfg):
    """
        For every dataset, we change the train_mask, test_mask, val_mask to a list of masks train_masks, test_masks, val_masks
        For node-level datasets, we return a single Data object with corresponding masks
        For graph-level datasets, we return (train_dataset, val_dataset, test_dataset)
    """
    train_transform, test_transform = get_transforms_by_config(cfg)
    pre_train_transform, pre_test_transform = get_pre_transforms_by_config(cfg)
    if cfg.dataset.name.lower() == 'chameleon' or cfg.dataset.name.lower() == 'squirrel':
        # wiki = WikipediaNetwork(root=cfg.dataset.root, name=cfg.dataset.name.lower())
        if not osp.exists(cfg.dataset.root, cfg.dataset.name.lower()):
            print('Downloading {} dataset...'.format(cfg.dataset.name.lower()))
            r = requests.get(fixed_urls[cfg.dataset.name.lower()])
            with open(osp.join(cfg.dataset.root, cfg.dataset.name.lower() + '_filtered.npz'), 'wb') as f:
                f.write(r.content)
        data = np.load(osp.join(cfg.dataset.root, cfg.dataset.name.lower() + '_filtered.npz'))
        features = torch.from_numpy(data['node_features'])
        labels = torch.from_numpy(data['node_labels'])
        edge_index = torch.from_numpy(data['edges'].T)
        train_masks = torch.from_numpy(data['train_masks'])
        test_masks = torch.from_numpy(data['test_masks'])
        val_masks = torch.from_numpy(data['val_masks'])
        data = Data(x=features, y=labels, edge_index=edge_index, train_masks=train_masks, test_masks=test_masks, val_masks=val_masks)
        if pre_train_transform is not None:
            data = pre_train_transform(data)
        if train_transform is not None:
            data = train_transform(data)
        return data
    elif cfg.dataset.name.lower() == 'cora' or cfg.dataset.name.lower() == 'citeseer' or cfg.dataset.name.lower() == 'pubmed':
        dataset = Planetoid(root=cfg.dataset.root, name=cfg.dataset.name.lower(), transform=train_transform, pre_transform=pre_train_transform)
        data = dataset[0]
        if not cfg.dataset.planetoid_high:
            data.train_masks = [data.train_mask]
            data.test_masks = [data.test_mask]
            data.val_masks = [data.val_mask]
        else:
            train_masks, val_masks, test_masks = get_random_split_masks(data.x.shape[0], 0.6, 0.2, cfg.seeds)
            data.train_masks = train_masks
            data.test_masks = test_masks
            data.val_masks = val_masks
        del data.train_mask, data.test_mask, data.val_mask
        return data
    elif 'ogbg' in cfg.dataset.name.lower():
        ## graph-level OGB datasets
        ogb_data = PygGraphPropPredDataset(name=cfg.dataset.name.lower(), root=cfg.dataset.root, pre_transform=pre_train_transform)
        split_idx = ogb_data.get_idx_split()
        # return ogb_data, split_idx['train'], split_idx['valid'], split_idx['test']
        train_data, val_data, test_data = ogb_data[split_idx['train']], ogb_data[split_idx['valid']], ogb_data[split_idx['test']]
        train_dataset.transform = train_transform
        val_dataset.transform = test_transform
        test_dataset.transform = test_transform
        return train_data, val_data, test_data
    elif 'ogbn' in cfg.dataset.name.lower():
        ogb_data = PygNodePropPredDataset(name=cfg.dataset.name.lower(), root=cfg.dataset.root, transform=train_transform, pre_transform=pre_train_transform)
        split_idx = ogb_data.get_idx_split()
        return ogb_data[0], split_idx['train'], split_idx['valid'], split_idx['test']
    elif cfg.dataset.name.lower() == 'actor':
        dataset = Actor(root=cfg.dataset.root, transform=train_transform, pre_transform=pre_train_transform)
        data = dataset[0]
        data.train_masks = [data.train_mask[:, i] for i in range(data.train_mask.shape[1])]
        data.test_masks = [data.test_mask[:, i] for i in range(data.test_mask.shape[1])]
        data.val_masks = [data.val_mask[:, i] for i in range(data.val_mask.shape[1])]
        del data.train_mask, data.test_mask, data.val_mask
        return data
    elif cfg.dataset.name.lower() == 'peptides-func' or cfg.dataset.name.lower() == 'peptides-struct':
        train_dataset = LRGBDataset(root=cfg.dataset.root, name=cfg.dataset.name.lower(), split='train', transform=train_transform, pre_transform=pre_train_transform)
        val_dataset = LRGBDataset(root=cfg.dataset.root, name=cfg.dataset.name.lower(), split='val', transform=test_transform, pre_transform=pre_test_transform)
        test_dataset = LRGBDataset(root=cfg.dataset.root, name=cfg.dataset.name.lower(), split='test', transform=test_transform, pre_transform=pre_test_transform)
        return train_dataset, val_dataset, test_dataset
    elif cfg.dataset.name.lower() == 'reddit-multi-12k':
        ## this one must be upper
        dataset = TUDataset(root=cfg.dataset.root, name=cfg.dataset.name.upper(), cleaned=True, pre_transform=pre_train_transform)
        train_mask, val_mask, test_mask = generate_data_masks_torch(0.8, 0.1, len(dataset))
        train_data, val_data, test_data = dataset[train_mask], dataset[val_mask], dataset[test_mask]
        train_data.transform = train_transform
        val_data.transform = test_transform
        test_data.transform = test_transform
        return train_data, val_data, test_data
    elif cfg.dataset.name.lower() == 'pokec':
        data = load_pokec_mat(cfg.dataset.root)
        return data
    else:
        print('Dataset {} not supported'.format(cfg.dataset.name.lower()))
        raise NotImplementedError
