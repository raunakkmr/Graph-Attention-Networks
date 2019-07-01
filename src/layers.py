import math

import numpy as np
import torch
import torch.nn as nn

class GraphAttention(nn.Module):

    def __init__(self, input_dim, output_dim, num_heads, dropout=0.5):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input node features.
        output_dim : int
            Dimension of output features after each attention head.
        num_heads : int
            Number of attention heads.
        dropout : float
            Dropout rate. Default: 0.5.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads

        self.fcs = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_heads)])
        self.a = nn.ModuleList([nn.Linear(2*output_dim, 1) for _ in range(num_heads)])

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=0)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, features, nodes, mapping, rows):
        """
        Parameters
        ----------
        features : torch.Tensor
            An (n' x input_dim) tensor of input node features.
        nodes : numpy array
            nodes is a numpy array of nodes in the current layer of the computation graph.
        mapping : dict
            mapping is a dictionary mapping node v (labelled 0 to |V|-1) to
            its position in the layer of nodes in the computation graph
            before nodes. For example, if the layer before nodes is [2,5],
            then mapping[2] = 0 and mapping[5] = 1.
        rows : numpy array
            rows[i] is an array of neighbors of node i which is present in nodes.

        Returns
        -------
        out : list of torch.Tensor
            A list of (len(nodes) x input_dim) tensor of output node features.
        """

        nprime = features.shape[0]
        rows = [np.array([mapping[v] for v in row], dtype=np.int64) for row in rows]
        sum_degs = np.hstack(([0], np.cumsum([len(row) for row in rows])))
        mapped_nodes = [mapping[v] for v in nodes]
        indices = torch.LongTensor([[v, c] for (v, row) in zip(mapped_nodes, rows) for c in row]).t()
        # indices = torch.LongTensor([[mapping[nodes[i]], c] for i in range(len(rows)) for c in rows[i]]).t()

        out = []
        for k in range(self.num_heads):
            h = self.fcs[k](features)

            nbr_h = torch.cat(tuple([h[row] for row in rows]), dim=0)
            self_h = torch.cat(tuple([h[mapping[nodes[i]]].repeat(len(row), 1) for (i, row) in enumerate(rows)]), dim=0)
            cat_h = torch.cat((self_h, nbr_h), dim=1)

            e = self.leakyrelu(self.a[k](cat_h))

            alpha = [self.softmax(e[lo : hi]) for (lo, hi) in zip(sum_degs, sum_degs[1:])]
            alpha = torch.cat(tuple(alpha), dim=0)
            alpha = alpha.squeeze(1)
            alpha = self.dropout(alpha)

            adj = torch.sparse.FloatTensor(indices, alpha, torch.Size([nprime, nprime]))
            out.append(torch.sparse.mm(adj, h)[mapped_nodes])

        return out