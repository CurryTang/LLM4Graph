import os
import argparse
from yacs.config import CfgNode as CN  

def set_config(cfg):
    ## Basic config
    cfg.seeds = [0]
    cfg.wandb_enable = False
    cfg.wandb_project = "LLM4Graph"
    cfg.wandb = "you-will-never-know"
    cfg.exp_name = "test"
    cfg.num_workers = 8
    cfg.device = 'cuda'
    cfg.n_trials = 30
    cfg.optuna_db = "sqlite:////egr/research-dselab/chenzh85/LLM4Graph/LLM4Graph/root/optuna.db"
    cfg.show_train_details = True

    ## dataset-related variables
    cfg.dataset = CN()
    cfg.dataset.name = 'cora'
    cfg.dataset.level = 'node'
    cfg.dataset.root = "/egr/research-dselab/chenzh85/LLM4Graph/LLM4Graph/root"
    cfg.dataset.re_generate_random_mask = False
    ## only valid if re_generate_random_mask is True
    cfg.dataset.train_ratio = -1
    cfg.dataset.val_ratio = -1
    cfg.dataset.test_ratio = -1
    cfg.dataset.planetoid_high = False
    ### only need to be set if dataset is not from ogb
    cfg.dataset.eval = 'accuracy'
    cfg.dataset.loss = 'cross-entropy'
    ## choices: saint, torch, pyg, sage
    cfg.dataset.loader = 'saint'
    ## whether we want to maximize the objective (classification) or minimize (regression)
    cfg.dataset.objective = 'maximize'

    ## path-related variables
    cfg.logging = "/egr/research-dselab/chenzh85/LLM4Graph/LLM4Graph/logging"


    ## environment-related variables
    cfg.env = CN()
    cfg.env.llama_path = "/mnt/home/chenzh85/graphlang/Graph-LLM/llama2-7b"

    ## model-related configs
    cfg.model = CN()
    cfg.model.name = 'GCN'
    cfg.model.nlayer_gnn = 2
    cfg.model.nlayer_gt = 2
    cfg.model.nhead = 1
    cfg.model.hidden_dim = 64
    cfg.model.dropout = 0.5
    cfg.model.attention_dropout = 0.5
    cfg.model.act = 'relu'
    cfg.model.norm = 'batchNorm'
    ## set this when load the data 
    cfg.model.num_classes = -1
    cfg.model.num_features = -1
    ## for SGFormer
    cfg.model.graph_weight = 0.8
    ## for NAGPhormer
    cfg.model.feature_prop_hop = 3
    cfg.model.gt = CN() 
    cfg.model.gt.pe_dim = 16

    


    ## common training utils config
    cfg.train = CN()
    cfg.train.optim = 'adam'
    cfg.train.scheduler = None
    cfg.train.lr_reduce_factor = 0.5
    cfg.train.lr_schedule_patience = 20
    ## two ways to quit training: min_lr or early_stop
    cfg.train.min_lr = -1
    cfg.train.full_batch = False
    ## only valid if full_batch is False
    cfg.train.batch_size = 32
    cfg.train.eval_batch_size = 32
    cfg.train.num_epochs = 100
    cfg.train.lr = 5e-5
    cfg.train.weight_decay = 0.1
    cfg.train.dropout = 0.5 
    cfg.train.grad_steps = 2
    cfg.train.max_norm = 0.1
    cfg.train.warmup_epochs = 0
    cfg.train.grad_steps = 1
    cfg.train.early_stop = True
    cfg.train.early_stop_patience = 20

    return cfg




def update_cfg(cfg, args_str=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="",
                        metavar="FILE", help="Path to config file")
    # opts arg needs to match set_cfg
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    if isinstance(args_str, str):
        # parse from a string
        args = parser.parse_args(args_str.split())
    else:
        # parse from command line
        args = parser.parse_args()
    # Clone the original cfg
    cfg = cfg.clone()

    # Update from config file
    if os.path.isfile(args.config):
        cfg.merge_from_file(args.config)

    # Update from command line
    cfg.merge_from_list(args.opts)


    return cfg


def update_yacs_config(cfg, new_params: dict):
    """
    Update a YACS configuration object with new parameters from a dictionary.

    Args:
    cfg (CfgNode): The original YACS configuration object.
    new_params (dict): A dictionary of parameters to update in the cfg.

    Returns:
    CfgNode: The updated configuration object.
    """
    # Convert the dictionary to a list of strings in YACS format
    param_list = []
    for key, value in new_params.items():
        param_list.append(f'{key}')
        param_list.append(f'{value}')

    # Update the cfg using merge_from_list
    cfg.merge_from_list(param_list)

    return cfg



cfg = set_config(CN())