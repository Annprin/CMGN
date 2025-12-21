import os
os.environ["DGL_DISABLE_GRAPHBOLT"] = "1"

import dgl.function as fn
import torch.nn as nn
import torch.nn.init as init
from torch.nn import Dropout, LayerNorm
import torch.nn.functional as F
from typing import List


# =========================
# GCN message passing
# =========================

gcn_msg = fn.copy_u('h', 'm')
gcn_reduce_mean = fn.mean(msg='m', out='h')


# =========================
# Simple FeedForward (AllenNLP replacement)
# =========================

class SimpleFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

    def get_output_dim(self):
        return self.linear2.out_features


# =========================
# GCN layer
# =========================

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation, init_range):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
        self._init_weights(init_range)

    def _init_weights(self, init_range):
        for param in self.linear.parameters():
            if param.dim() >= 2:
                init.xavier_uniform_(param)
            else:
                init.zeros_(param)

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h': h}


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation, init_range):
        super().__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation, init_range)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce_mean)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')


# =========================
# GCN Network
# =========================

class GCNNet(nn.Module):
    def __init__(self, hdim, init_range, nlayers=2, dropout_prob=0.1):
        super().__init__()

        self.gcn_layers = nn.ModuleList()
        self.feedforward_layers = nn.ModuleList()
        self.ff_layer_norms = nn.ModuleList()
        self.gcn_layer_norms = nn.ModuleList()

        for _ in range(nlayers):
            ff = SimpleFeedForward(
                input_dim=hdim,
                hidden_dim=hdim,
                output_dim=hdim,
                dropout=dropout_prob
            )
            self.feedforward_layers.append(ff)
            self.ff_layer_norms.append(LayerNorm(ff.get_output_dim()))

            gcn = GCN(hdim, hdim, F.relu, init_range)
            self.gcn_layers.append(gcn)
            self.gcn_layer_norms.append(LayerNorm(hdim))

        self.dropout = Dropout(dropout_prob)
        self._input_dim = hdim
        self._output_dim = hdim

    def forward(self, g, features):
        output = features

        for i in range(len(self.gcn_layers)):
            # FeedForward block
            ff_out = self.feedforward_layers[i](output)
            ff_out = self.dropout(ff_out)
            output = self.ff_layer_norms[i](ff_out + output)

            # GCN block
            gcn_out = self.gcn_layers[i](g, output)
            gcn_out = self.dropout(gcn_out)
            output = self.gcn_layer_norms[i](gcn_out + output)

        return output
