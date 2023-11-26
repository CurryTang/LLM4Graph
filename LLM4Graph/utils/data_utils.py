import torch
import numpy as np
from LLM4Graph.utils.misc import seed_everything
from LLM4Graph.utils.text_utils import get_tfidf_vector
from torch_geometric.utils import index_to_mask, degree, homophily
from torch_scatter import scatter_add


class LabelPerClassSplit(object):
    """
    Class for splitting data into training, validation, and test sets based on labels.

    This class provides a callable object for splitting data into training, validation, and test sets.
    The splitting is done based on the labels of the data, with a specified number of labels per class for the training set.
    """
    def __init__(
            self,
            num_labels_per_class: int = 20,
            num_valid: int = 500,
            num_test: int = -1,
            inside_old_mask: bool = False
    ):
        """
        Constructor method for the LabelPerClassSplit class.

        Initializes a new instance of the LabelPerClassSplit class with the provided parameters.

        Parameters:
        num_labels_per_class (int, optional): The number of labels per class for the training set. Defaults to 20.
        num_valid (int, optional): The number of validation data points. Defaults to 500.
        num_test (int, optional): The number of test data points. If -1, all remaining data points after training and validation are used for testing. Defaults to -1.
        inside_old_mask (bool, optional): Whether to consider only data points inside the old mask for splitting. Defaults to False.

        Returns:
        None
        """
        self.num_labels_per_class = num_labels_per_class
        self.num_valid = num_valid
        self.num_test = num_test
        self.inside_old_mask = inside_old_mask

    def __call__(self, data, total_num):
        """
        Callable method for the LabelPerClassSplit class.

        This method splits the data into training, validation, and test sets based on the labels of the data.

        Parameters:
        data: The data to be split.
        total_num (int): The total number of data points.

        Returns:
        tuple: A tuple containing the masks for the training, validation, and test sets.
        """
        new_train_mask = torch.zeros(total_num, dtype=torch.bool)
        new_val_mask = torch.zeros(total_num, dtype=torch.bool)
        new_test_mask = torch.zeros(total_num, dtype=torch.bool)

        perm = torch.randperm(total_num)
        train_cnt = np.zeros(data.y.max().item() + 1, dtype=np.int32)

        for i in range(perm.numel()):
            label = data.y[perm[i]]
            if train_cnt[label] < self.num_labels_per_class:
                train_cnt[label] += 1
                new_train_mask[perm[i]] = 1
            elif new_val_mask.sum() < self.num_valid:
                new_val_mask[perm[i]] = 1
            else:
                if self.num_test != -1:
                    if new_test_mask.sum() < self.num_test:
                        new_test_mask[perm[i]] = 1
                    else:
                        new_test_mask[perm[i]] = 1

        
        return new_train_mask, new_val_mask, new_test_mask

def generate_random_mask(total_node_number, train_num, val_num, test_num = -1, seed = 0):
    """
    Function to generate random masks for training, validation, and test sets.

    This function generates random masks for training, validation, and test sets based on the provided numbers of nodes for each set.
    The randomness is controlled by a seed, which can be set for reproducibility.

    Parameters:
    total_node_number (int): The total number of nodes.
    train_num (int): The number of nodes in the training set.
    val_num (int): The number of nodes in the validation set.
    test_num (int, optional): The number of nodes in the test set. If -1, all remaining nodes after training and validation are used for testing. Defaults to -1.
    seed (int, optional): The seed for the random number generator. Defaults to 0.

    Returns:
    tuple: A tuple containing the masks for the training, validation, and test sets.
    """
    seed_everything(seed)
    random_index = torch.randperm(total_node_number)
    train_index = random_index[:train_num]
    val_index = random_index[train_num:train_num + val_num]
    if test_num == -1:
        test_index = random_index[train_num + val_num:]
    else:
        test_index = random_index[train_num + val_num: train_num + val_num + test_num]
    return index_to_mask(train_index, total_node_number), index_to_mask(val_index, total_node_number), index_to_mask(test_index, total_node_number)


def get_mask(data, mode, train_ratio = 0.6, val_ratio = 0.2, test_ratio = 0.2, seeds = [0, 1, 2], texts = None, homo_split = 0.5, 
             max_df = 1., min_df = 1, max_features = None):
    """
        Mask generation function
        * semi: 20 per class-500-1000 split for cora, citeseer, pubmed
        * high: 60%-20%-20% split for cora, citeseer, pubmed
        * random: random split with given train, val, test
        * ood_degree: split the data based on the degree of the nodes
        * ood_homo: split the data based on homophily
        * ood_concept: split the data based on node attributes
    """
    train_masks = []
    val_masks = []
    test_masks = []
    for s in seeds:
        seed_everything(s)
        if mode == 'semi':
            spliter = LabelPerClassSplit()
            train_mask, val_mask, test_mask = spliter(data, data.x.shape[0])
        elif mode == 'high' or mode == 'random':
            if mode == 'random':
                train_num = int(data.x.shape[0] * train_ratio)
                val_num = int(data.x.shape[0] * val_ratio)
                if train_ratio + val_ratio + test_ratio == 1.:
                    test_num = data.x.shape[0] - train_num - val_num
                else:
                    test_num = int(data.x.shape[0] * test_ratio)
            else:
                train_num = int(data.x.shape[0] * .6)
                val_num = int(data.x.shape[0] * .2)
                test_num = data.x.shape[0] - train_num - val_num 
            train_mask, val_mask, test_mask = generate_random_mask(
                data.x.shape[0], train_num, val_num, test_num, seed=s
            )             
        elif mode == 'ood_degree':
            train_mask, val_mask, test_mask = get_ood_degree_split(
                data, train_ratio, val_ratio, test_ratio
            ) 
        elif mode == 'ood_homo':
            train_mask, val_mask, test_mask = get_ood_homo_split(
                data, train_ratio, val_ratio, test_ratio, homo_split
            )
        elif mode == 'ood_concept':
            train_mask, val_mask, test_mask = get_ood_word_split(
                data, train_ratio, val_ratio, texts, max_df, min_df, 
                max_features, test_ratio
            ) 
        train_masks.append(train_mask)
        val_masks.append(val_mask)
        test_masks.append(test_mask)
    return train_masks, val_masks, test_masks


def get_ood_degree_split(data, train_ratio, val_ratio, test_ratio):
    """
    Function to split data into training, validation, and test sets based on node degrees.

    This function splits the data into training, validation, and test sets based on the degrees of the nodes.
    The nodes with lower degrees are more likely to be included in the training set, while the nodes with higher degrees are more likely to be included in the test set.
    The ratios for the training, validation, and test sets are provided as parameters.

    Parameters:
    data: The data to be split. The data should have an attribute `x` representing the node features and an attribute `edge_index` representing the graph structure.
    train_ratio (float): The ratio of nodes to include in the training set.
    val_ratio (float): The ratio of nodes to include in the validation set.
    test_ratio (float): The ratio of nodes to include in the test set.

    Returns:
    tuple: A tuple containing the masks for the training, validation, and test sets.
    """
    total_node_number = data.x.shape[0]
    train_num = int(total_node_number * train_ratio)
    val_num = int(total_node_number * val_ratio)
    if train_ratio + val_ratio + test_ratio == 1.:
        test_num = -1
    else:
        test_num = int(data.x.shape[0] * test_ratio)
    node_degrees = degree(data.edge_index[0])
    _, sorted_index = torch.sort(node_degrees)
    train_index = sorted_index[:train_num]
    val_index = sorted_index[train_num:train_num + val_num]
    if test_num == -1:
        test_index = sorted_index[train_num + val_num:]
    else:
        test_index = sorted_index[train_num + val_num: train_num + val_num + test_num]
    return index_to_mask(train_index, total_node_number), index_to_mask(val_index, total_node_number), index_to_mask(test_index, total_node_number)



def node_level_homophily(edge_index, node_labels):
    """
        Pyg only offers a graph-level homophily, this functions computes the node-level homophily

        Parameters:
            edge_index: edge index of the graph
            node_labels: node labels of the graph
        
        Returns:
            node_level_homophily: node-level homophily
    """
    row, col = edge_index[0], edge_index[1]
    edge_values = torch.ones(edge_index.shape[1])
    deg = scatter_add(edge_values, row, dim = 0, dim_size = node_labels.shape[0])

    edge_homo_value = (node_labels[row] == node_labels[col]).int()
    homo_ratio = scatter_add(
        edge_homo_value, row, dim = 0, dim_size=node_labels.shape[0]
    )

    homo_ratio = torch.squeeze(homo_ratio)
    return homo_ratio / deg



def get_ood_homo_split(data, homo_split = 0.5):
    """
    Function to split data into training, validation, and test sets based on node homophily.

    This function splits the data into training, validation, and test sets based on the homophily of the nodes.
    The nodes with lower homophily are more likely to be included in the training set, while the nodes with higher homophily are more likely to be included in the test set.
    The ratios for the training, validation, and test sets are provided as parameters.

    Check Paper:
    Demystifying Structural Disparity in Graph Neural Networks: Can One Size Fit All?

    Parameters:
    data: The data to be split. The data should have an attribute `x` representing the node features and an attribute `edge_index` representing the graph structure.
    train_ratio (float): The ratio of nodes to include in the training set.
    val_ratio (float): The ratio of nodes to include in the validation set.
    test_ratio (float): The ratio of nodes to include in the test set.

    Returns:
    tuple: A tuple containing the masks for the training, validation, and test sets.
    """
    ## get average homophily to determine whether the majority of nodes are homophilic or heterophilic
    avg_homo = homophily(data.edge_index, data.y, method='node')
    node_homo = node_level_homophily(data.edge_index, data.y)
    if avg_homo < homo_split:
        majority_mask = node_homo < homo_split
    else:
        majority_mask = node_homo > homo_split
    minority_mask = ~majority_mask

    ## select 20% from the majority as the validation mask
    true_indices = torch.nonzero(majority_mask, as_tuple=False).squeeze()
    shuffle_indices = true_indices[torch.randperm(true_indices.size(0))]
    selected_indices = shuffle_indices[:int(0.2 * len(shuffle_indices))]
    val_mask = torch.zeros_like(majority_mask, dtype=torch.bool)
    val_mask[selected_indices] = True 
    train_mask = ~val_mask & majority_mask
    test_mask = minority_mask

    return train_mask, val_mask, test_mask


def get_ood_word_split(data, train_ratio, val_ratio, text_df, max_df, min_df, max_features, test_ratio = 0):
    total_node_number = data.x.shape[0]
    train_num = int(total_node_number * train_ratio)
    val_num = int(total_node_number * val_ratio)
    if train_ratio + val_ratio + test_ratio == 1.:
        test_num = -1
    else:
        test_num = int(data.x.shape[0] * test_ratio)
    texts = text_df.apply(
        lambda row: row['title'] + ' ' + row['content'], axis = 1
    ).tolist()

    tfidf_X = get_tfidf_vector(texts, max_df=max_df, min_df=min_df, max_features=max_features)
    tfidf_sum = tfidf_X.sum(-1)

    _, sorted_index = torch.sort(tfidf_sum)
    train_index = sorted_index[:train_num]
    val_index = sorted_index[train_num:train_num + val_num]
    if test_num == -1:
        test_index = sorted_index[train_num + val_num:]
    else:
        test_index = sorted_index[train_num + val_num: train_num + val_num + test_num]
    return index_to_mask(train_index, total_node_number), index_to_mask(val_index, total_node_number), index_to_mask(test_index, total_node_number)








    