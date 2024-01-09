import os
import argparse
from yacs.config import CfgNode as CN  

def set_config(cfg):
    ## Basic config
    cfg.seed = 0
    cfg.wandb = "you-will-never-know"
    cfg.num_workers = 8
    cfg.device = 'cuda'

    ## dataset-related variables
    cfg.dataset = CN()
    cfg.dataset.name = 'cora'
    cfg.dataset.root = "/mnt/home/chenzh85/graphlang/LLM4Graph/root"
    cfg.dataset.re_generate_random_mask = False
    ## only valid if re_generate_random_mask is True
    cfg.dataset.train_ratio = -1
    cfg.dataset.val_ratio = -1
    cfg.dataset.test_ratio = -1

    ## path-related variables
    cfg.logging = "/mnt/home/chenzh85/graphlang/LLM4Graph/logging"


    ## environment-related variables
    cfg.env = CN()
    cfg.env.llama_path = "/mnt/home/chenzh85/graphlang/Graph-LLM/llama2-7b"
    cfg.env.metis_dll = "/mnt/home/chenzh85/anaconda3/envs/acl24/lib/libmetis.so"
    cfg.optim = 'adam'

    ## common training utils config
    cfg.train = CN()
    cfg.train.optim = 'adam'
    cfg.train.scheduler = None
    cfg.train.lr_reduce_factor = 0.5
    cfg.train.lr_schedule_patience = 20
    cfg.train.min_lr = 1e-5
    cfg.train.full_batch = False
    ## only valid if full_batch is False
    cfg.train.batch_size = 32
    cfg.train.eval_batch_size = 32
    cfg.train.sampler = 'saint'
    cfg.train.num_epochs = 20
    cfg.train.lr = 5e-5
    cfg.train.weight_decay = 0.1
    cfg.train.dropout = 0.5 
    cfg.train.grad_steps = 2
    cfg.train.max_norm = 0.1
    cfg.train.warmup_epochs = 0
    cfg.train.grad_steps = 1
    cfg.train.early_stop = True
    cfg.train.early_stop_patience = 20


    ## graph transformer specific config
    cfg.gt = CN()
    cfg.gt.class_token = False 
    cfg.gt.global_pool = 'avg'
    cfg.gt.task_type = 'regression'
    cfg.gt.attn_type = 'performer'
    # cfg.gt.gt_type = 'MLPMixer'
    cfg.gt.gnn_type = 'GINEConv'
    cfg.gt.fc_norm = None 
    cfg.gt.norm_layer = None
    cfg.gt.act_layer = None 
    cfg.gt.nlayer_gt = 4
    cfg.gt.nlayer_gnn = 4
    cfg.gt.nlayer_mixer = 4
    cfg.gt.bn = True
    cfg.gt.res = True 
    cfg.gt.metis = CN()
    cfg.gt.metis.n_patches = 32
    cfg.gt.metis.num_hops = 1
    cfg.gt.metis.drop_rate = 0.0
    cfg.gt.metis.online = True
    cfg.gt.pos_enc = CN()
    cfg.gt.pos_enc.lap_dim = 8
    cfg.gt.pos_enc.rw_dim = 0
    cfg.gt.pos_enc.patch_num_diff = -1
    cfg.gt.pos_enc.patch_rw_dim = 8
    cfg.gt.node_type = 'Discrete'
    cfg.gt.edge_type = 'Discrete'
    cfg.gt.nout = 1
    cfg.gt.nfeat_node = 28
    cfg.gt.nfeat_edge = 4
    cfg.gt.channel = 64
    cfg.gt.redraw = True


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


cfg = set_config(CN())