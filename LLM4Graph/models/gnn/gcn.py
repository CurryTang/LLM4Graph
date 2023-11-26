import torch 
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dimension, num_classes, dropout, norm=None) -> None:
        super().__init__()
        assert norm in [None, 'batchNorm', 'layerNorm']
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout
        if num_layers == 1:
            self.convs.append(GCNConv(input_dim, num_classes, cached=False,
                             normalize=True))
        else:
            self.convs.append(GCNConv(input_dim, hidden_dimension, cached=False,
                             normalize=True))
            if norm:
                if norm == 'batchNorm':
                    self.norms.append(torch.nn.BatchNorm1d(hidden_dimension))
                else:
                    self.norms.append(torch.nn.LayerNorm(hidden_dimension))
            else:
                self.norms.append(torch.nn.Identity())

            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dimension, hidden_dimension, cached=False,
                             normalize=True))
                if norm:
                    if norm == 'batchNorm':
                        self.norms.append(torch.nn.BatchNorm1d(hidden_dimension))
                    else:
                        self.norms.append(torch.nn.LayerNorm(hidden_dimension))
                else:
                    self.norm.append(torch.nn.Identity())

            self.convs.append(GCNConv(hidden_dimension, num_classes, cached=False, normalize=True))

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[i](x, edge_index, edge_weight)
            if i != self.num_layers - 1:
                x = self.norms[i](x)
                x = F.relu(x)
        return x