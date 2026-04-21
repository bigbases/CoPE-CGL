import torch
import torch.nn.functional as F
from torch_sparse import matmul


class SGCNeighborAgg(torch.nn.Module):
    def __init__(self, k=2):
        super().__init__()
        self.k = k

    def forward(self, x, adj_t):
        deg = adj_t.sum(dim=1).to(torch.float).clamp(min=1)
        norm = deg.pow(-0.5).unsqueeze(1)

        for _ in range(self.k):
            x = x * norm
            x = matmul(adj_t, x)
            x = x * norm
        return x


class TAAMSGC(torch.nn.Module):
    def __init__(self, nin, h_dims, k=2, dropout=0.0, batch_norm=False, linear_bias=True):
        super().__init__()
        if len(h_dims) == 0:
            raise ValueError("TAAMSGC requires at least one hidden dimension.")

        self.neighbor_agg = SGCNeighborAgg(k=k)
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.relu = torch.nn.LeakyReLU(0.2)
        self.out_dim = h_dims[-1]

        dims = [nin] + h_dims + [h_dims[-1]]
        self.feat_trans_layers = torch.nn.ModuleList([])
        self.bns = torch.nn.ModuleList([])

        for i in range(len(dims) - 1):
            self.feat_trans_layers.append(
                torch.nn.Linear(dims[i], dims[i + 1], bias=linear_bias)
            )
            if self.batch_norm and i < len(dims) - 2:
                self.bns.append(torch.nn.BatchNorm1d(dims[i + 1]))

    def feat_trans(self, x):
        for i, layer in enumerate(self.feat_trans_layers[:-1]):
            x = layer(x)
            if self.batch_norm:
                x = self.bns[i](x)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.feat_trans_layers[-1](x)
        return x

    def forward(self, data):
        x = self.neighbor_agg(data.x, data.adj_t)
        return self.feat_trans(x)

    def reset_parameters(self):
        for layer in self.feat_trans_layers:
            layer.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
