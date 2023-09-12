import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class NodePredictorHead(nn.Module):
    def __init__(self, latent_dim=256, sec_dim=64, dropout=0.2):
        super(NodePredictorHead, self).__init__()
        self.fc1 = nn.Linear(latent_dim, sec_dim)
        self.fc2 = nn.Linear(sec_dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        return x


class EdgePredictor_HadamandHead(nn.Module):
    def __init__(self, num_nodes, latent_dim=256, sec_dim=512, dropout=0.2):
        super(EdgePredictor_HadamandHead, self).__init__()
        self.fc1 = nn.Linear(latent_dim, sec_dim)
        self.drop = nn.Dropout(dropout)
        self.num_nodes = num_nodes
        self.fc2 = nn.Linear(sec_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        T, num_nodes, embed_size = x.shape
        # trick for matricizing hadamand product
        edge_embed = torch.vstack([xi.view(num_nodes, 1, embed_size) * xi for xi in x]) 
        edge_embed = edge_embed.view(T, num_nodes, num_nodes, embed_size)
        return self.fc2(edge_embed)

    
class EdgePredictor_EuclideanDistanceHead(nn.Module):
    def __init__(self, num_nodes, latent_dim=256, sec_dim=512, dropout=0.2):
        super(EdgePredictor_EuclideanDistanceHead, self).__init__()
        self.fc1 = nn.Linear(latent_dim, sec_dim)
        self.drop = nn.Dropout(dropout)
        self.num_nodes = num_nodes
        self.fc2 = nn.Parameter(torch.randn(num_nodes, num_nodes, requires_grad=True))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        edge_embed = torch.cdist(x, x)
        return self.fc2 * edge_embed


class EdgePredictor_SumHead(nn.Module):
    def __init__(self, num_nodes, latent_dim=256, sec_dim=512, third_dim=1024, dropout=0.2):
        super(EdgePredictor_SumHead, self).__init__()
        self.num_nodes = num_nodes
        self.fc1 = nn.Linear(latent_dim, sec_dim)
        self.fc2 = nn.Linear(sec_dim, third_dim)
        self.fc3 = nn.Linear(third_dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        T, num_nodes, embed_size = x.shape
        # trick for matricizing self-summation
        edge_embed = torch.vstack([xi.view(num_nodes, 1, embed_size) + xi for xi in x]) 
        edge_embed = edge_embed.view(T, num_nodes, num_nodes, embed_size)
        return self.fc3(edge_embed)


# https://github.com/pytorch/examples/blob/main/word_language_model/main.py
class ReTrend_LSTM(nn.Module):
    def __init__(self, num_nodes, hidden_dim=256, nlayers=10, dropout=0.2):
        super(ReTrend_LSTM, self).__init__()
        self.embed_table = nn.Embedding(num_nodes, 768)
        self.lstm = nn.LSTM(768, hidden_dim, nlayers, dropout=dropout)
        self.predictor = NodePredictorHead()
        self.nhid = hidden_dim
        self.nlayers = nlayers
        self.num_nodes = num_nodes

    def repackage_hidden(self, h):
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def init_embed(self, embed_matrix):
        self.embed_table.weight.data.copy_(embed_matrix.data)
       
    def forward(self, hidden):
        hidden = self.repackage_hidden(hidden)
        T = hidden[0].shape[0]
        embed = self.embed_table(torch.arange(self.num_nodes).cuda()).repeat(T, 1).view(T, self.num_nodes, 768)        
        y, hidden = self.lstm(embed, hidden)
        y = self.predictor(y)
        return y, hidden

    def init_hidden(self, *args, **kw):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, self.num_nodes, self.nhid),
                    weight.new_zeros(self.nlayers, self.num_nodes, self.nhid))


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model=256, dropout=0.2, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

    
class TransformerModel(nn.Module):
    def __init__(self, max_len=10, ninp=256, nhead=8, nhid=256, nlayers=2, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout, max_len=max_len)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, node_embed, has_mask=True):
        if has_mask:
            device = node_embed.device
            if self.src_mask is None or self.src_mask.size(0) != len(node_embed):
                mask = self._generate_square_subsequent_mask(len(node_embed)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None
        node_embed = self.pos_encoder(node_embed)
        output = self.transformer_encoder(node_embed, self.src_mask)
        return output
