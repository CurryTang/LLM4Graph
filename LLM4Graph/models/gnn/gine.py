from torch_geometric.nn.models import GIN
import torch.nn as nn

class GINE(nn.Module):
    def __init__(self, in_channel, hidden_channel, num_layers, num_classes, dropout, norm = 'batchNorm') -> None:
        super().__init__()
        self.gnn = GIN(
            in_channel, 
            hidden_channel, 
            num_layers, 
            num_classes,
            dropout, 
            norm = norm
        )

    def forward(self, batch):
        return self.gnn(batch.x, batch.edge_index, batch.edge_attr)