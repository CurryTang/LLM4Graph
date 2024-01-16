from LLM4Graph.models.gnn import *
from LLM4Graph.models.graphTrans import *

def get_model(cfg):
    if cfg.model.name == 'GCN':
        return GCN(
            cfg.model.nlayer_gnn,
            cfg.model.num_features,
            cfg.model.hidden_dim,
            cfg.model.num_classes,
            cfg.model.dropout,
            cfg.model.norm
        )
    elif cfg.model.name == 'SGFormer':
        return SGFormer(
            cfg.model.num_features, 
            cfg.model.hidden_dim,
            cfg.model.num_classes,
            cfg.model.nlayer_gt, 
            cfg.model.nhead, 
            cfg.model.dropout,
            cfg.model.nlayer_gnn,
            cfg.model.dropout, 
            cfg.model.graph_weight
        )
    elif cfg.model.name == 'NAGPhormer':
        return NAGphormer(
            cfg.model.feature_prop_hop, 
            cfg.model.num_classes,
            cfg.model.num_features,
            cfg.model.gt.pe_dim,
            cfg.model.nlayer_gt,
            cfg.model.nhead,
            cfg.model.hidden_dim,
            cfg.model.hidden_dim,
            cfg.model.dropout,
            cfg.model.attention_dropout
        )
    

