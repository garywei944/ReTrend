import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, LEConv, ASAPooling
from torch_sparse import SparseTensor
from basic_model import *


class NodeEncoder(nn.Module):
    def __init__(self, embed_dim=768, latent_dim=256, dropout=0.2):
        super(NodeEncoder, self).__init__()
        self.gc1 = LEConv(embed_dim, latent_dim)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x, E_idx, E_w):
        x = F.relu(self.gc1(x, E_idx, E_w))
        x = self.drop(x)
        return x


class TimeSeriesEncoder(nn.Module):
    def __init__(self, nhid=256, nlayers=10, dropout=0.2):
        super(TimeSeriesEncoder, self).__init__()
        self.lstm = nn.LSTM(nhid, nhid, nlayers, dropout=dropout)
        self.nlayers = nlayers
        self.nhid = nhid
        
    def forward(self, x, h):
        return self.lstm(x, h)

    def init_hidden(self, num_node):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, num_node, self.nhid),
                    weight.new_zeros(self.nlayers, num_node, self.nhid))

    
class ReTrend(nn.Module):
    def __init__(self, num_nodes) -> None:
        super(ReTrend, self).__init__()
        self.num_nodes = num_nodes
        self.node_embed = nn.Embedding(num_nodes, 768)
        self.node_encoder = NodeEncoder()
        # T, K, 256 (LSTM, Transformer)
        self.time_series_encoder = TimeSeriesEncoder()
        self.node_decoder = NodePredictorHead()
    
    def repackage_hidden(self, h):
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def init_embed(self, embed_matrix):
        self.node_embed.weight.data.copy_(embed_matrix.data)
        
    def init_hidden(self, num_node):
        return self.time_series_encoder.init_hidden(num_node)

    def forward(self, E_indices, E_weights, hidden):
        x_embeds = []
        embed = self.node_embed(torch.arange(self.num_nodes).cuda())
        hidden = self.repackage_hidden(hidden)
        T = len(E_indices)
        for E_idx, E_w in zip(E_indices, E_weights):
            x_embed = self.node_encoder(embed, E_idx, E_w)
            x_embeds.append(x_embed)
        x_embeds = torch.vstack(x_embeds).view(T, self.num_nodes, 256)
        y, hidden = self.time_series_encoder(x_embeds, hidden)
        return self.node_decoder(y), hidden

    
class ReTrend_Transformer(nn.Module):
    def __init__(self, num_nodes, T, node_pred=True) -> None:
        super(ReTrend_Transformer, self).__init__()
        self.num_nodes = num_nodes
        self.latent_dim = 256
        self.node_embed = nn.Embedding(num_nodes, 768)
        self.node_encoder = NodeEncoder(latent_dim=self.latent_dim)
        self.time_series_encoder = TransformerModel(ninp=self.latent_dim, max_len=T)
        self.predictor = NodePredictorHead(latent_dim=self.latent_dim) if node_pred else EdgePredictor_SumHead(num_nodes)

    def init_embed(self, embed_matrix):
        self.node_embed.weight.data.copy_(embed_matrix.data)

    def forward(self, E_indices, E_weights):
        x_embeds = []
        embed = self.node_embed(torch.arange(self.num_nodes).cuda())
        T = len(E_indices)
        for E_idx, E_w in zip(E_indices, E_weights):
            x_embed = self.node_encoder(embed, E_idx, E_w)
            x_embeds.append(x_embed)
        x_embeds = torch.vstack(x_embeds).view(T, self.num_nodes, self.latent_dim)
        y = self.time_series_encoder(x_embeds, has_mask=False)
        return self.predictor(y)

    
class ReTrend_LastYearPrediction(nn.Module):
    def __init__(self, *args, **kw) -> None:
        super(ReTrend_LastYearPrediction, self).__init__()
        pass

    def forward(self, unshifted_nodes):
        return unshifted_nodes[-1]


class ReTrend_PolynomialRegression(nn.Module):
    def __init__(self, num_nodes, deg=3, **kw) -> None:
        super(ReTrend_PolynomialRegression, self).__init__()
        self.deg = deg
        self.weights_list = nn.ModuleList() 
        for _ in range(deg + 1):
            # include bias
            self.weights_list.add_module(
                nn.Parameter(torch.randn(num_nodes)) # polynomial weights
            )
        
    def forward(self, unshifted_nodes):
        predicted = torch.zeros_like(unshifted_nodes) + self.weights_list[0]
        T = len(unshifted_nodes)
        for d, w in enumerate(self.weights_list):
            for j in range(1, T + 1):
                # shift by one in prediction
                predicted[j] += w * j ** (d + 1)
        return predicted
