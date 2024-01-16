import torch.nn as nn
from LLM4Graph.models.elements import Identity
import LLM4Graph.models.elements.gnn_wrapper as gnn_wrapper
import torch.nn.functional as F

class GNN(nn.Module):
    def __init__(self,
                 nin,
                 nout,
                 nlayer_gnn,
                 gnn_type,
                 bn=True,
                 dropout=0.0,
                 res=True):
        """
            GNN architecture for Graph-ViT, it should be noted
            that nhidden is not given in this version
        """
        super().__init__()
        self.dropout = dropout
        self.res = res

        self.convs = nn.ModuleList([getattr(gnn_wrapper, gnn_type)(
            nin, nin, bias=not bn) for _ in range(nlayer_gnn)])
        self.norms = nn.ModuleList(
            [nn.BatchNorm1d(nin) if bn else Identity() for _ in range(nlayer_gnn)])
        self.output_encoder = nn.Linear(nin, nout)

    def forward(self, x, edge_index, edge_attr):
        previous_x = x
        for layer, norm in zip(self.convs, self.norms):
            x = layer(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            if self.res:
                x = x + previous_x
                previous_x = x

        x = self.output_encoder(x)
        return x