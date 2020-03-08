import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
import reformer
import util

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable


"""
CS224N course project model implementation: Transformer
"""

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=0.0):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    # (Dropout described below)
    p_attn = F.dropout(p_attn, p=dropout)
    return torch.matmul(p_attn, value), p_attn

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity we apply the norm first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer function that maintains the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # Torch linears have a `b` by default. 
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
        
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.p = dropout
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.p)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class EncoderLayer(nn.Module):
    """
    Encoder is made up of two sublayers, self-attn and feed forward (defined below)
    b blocks of cnn sublayers, each with c Conv1d 
    """

    # N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
    def __init__(self, size=512, d_ff=2048, h=8, dropout=0.1, kernel = 7, c = 4):
        super(EncoderLayer, self).__init__()
        self.c = c
        
        self.conv1d = nn.Sequential(
            nn.Conv1d(size, size, kernel, bias=True, padding=kernel//2),
            nn.ReLU()
        )
            
        self.self_attn = MultiHeadedAttention(h, size, dropout)
        self.feed_forward = PositionwiseFeedForward(size, d_ff, dropout)
        self.sublayer = clones(SublayerConnection(size, dropout), self.c + 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."

        # convolution
        for i in range(self.c):
            x = self.conv1d(x.transpose(1,2))
            #x = torch.max(x, dim=2)
            x = x.transpose(1,2)
            x = self.sublayer[i](x, lambda x: x)
        
        x = self.sublayer[self.c](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[self.c+1](x, self.feed_forward)

class TransformerEncoder(nn.Module):
    """
    The transformer encoder part described in 'Attention is all you need'
    b blocks of cnn sublayers, each with c Conv1d 
    """
    def __init__(self, hidden_size, N = 1, c = 4):
        super(TransformerEncoder, self).__init__()
        self.layer = EncoderLayer(size = hidden_size, c = c)
        self.layers = clones(self.layer, N)
        self.norm = LayerNorm(self.layer.size)

    def forward(self, x, mask):
        """
        Pass the input (and mask) through each layer in turn.
        """

        # haroldmei
        mask = torch.unsqueeze(mask, 1)

        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# just copied from BiDAFOutput
class Transformer_Output(nn.Module):
    """
    """
    def __init__(self, hidden_size, drop_prob):
        super(Transformer_Output, self).__init__()

        self.transformer = TransformerEncoder(hidden_size, N = 3) 
                              
        self.att_linear_1 = nn.Linear(4 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(hidden_size, 1)

        self.att_linear_2 = nn.Linear(4 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.transformer(mod, mask)
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = util.masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = util.masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2


class TransformerEncoderLayerEx(nn.Module):
    def __init__(self, d_model, dropout=0.1, c = 4, kernel = 7):
        super(TransformerEncoderLayerEx, self).__init__()
        self.c = c

        self.conv1d = [nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel, bias=True, padding=kernel//2).cuda(),
            nn.ReLU()
        )] * self.c

        self.norm3 = [nn.modules.transformer.LayerNorm(d_model)] * self.c
        self.dropout3 = [nn.modules.transformer.Dropout(dropout)] * self.c

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        for i in range(self.c):
            src2 = self.conv1d[i](src.transpose(1,2)).transpose(1,2)
            src = src + self.dropout3[i](src2)
            src = self.norm3[i](src)

        return src

# just copied from BiDAFOutput
class Transformer_OutputEx(nn.Module):
    """
    """
    def __init__(self, hidden_size, mod_layers, drop_prob):
        super(Transformer_OutputEx, self).__init__()
        self.cnn = TransformerEncoderLayerEx(hidden_size,c=2)
        self.transformer = nn.modules.transformer.TransformerEncoder(
            nn.modules.transformer.TransformerEncoderLayer(hidden_size, 8, dropout=drop_prob), 
            mod_layers, 
            nn.modules.transformer.LayerNorm(hidden_size)
        )
                              
        self.att_linear_1 = nn.Linear(4 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(hidden_size, 1)

        self.att_linear_2 = nn.Linear(4 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.transformer(self.cnn(mod)) #, mask)
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = util.masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = util.masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2



"""
CS224N course project model implementation: Reformer
"""
class ReformerEncoder(nn.Module):
    """
    The Reformer encoder part described in ''
    """
    def __init__(self, hidden_size, depth = 12, drop_prob=0.1, bucket_size = 16, max_seq_len=512):
        super(ReformerEncoder, self).__init__()
        self.reformer = reformer.Reformer(
            dim = hidden_size,
            depth = depth,
            bucket_size = bucket_size, 
            max_seq_len = max_seq_len,
            heads = 8,
            lsh_dropout = drop_prob,
            causal = False
        ).cuda()
        self.bucket_size = bucket_size

    def forward(self, x, mask):
        x = self.reformer(x)
        return x



# just copied from BiDAFOutput
class Reformer_Output(nn.Module):
    """
    """
    def __init__(self, hidden_size, drop_prob):
        super(Reformer_Output, self).__init__()

        self.transformer = ReformerEncoder(hidden_size, depth = 1) 
                              
        self.att_linear_1 = nn.Linear(4 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(hidden_size, 1)

        self.att_linear_2 = nn.Linear(4 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.transformer(mod, mask)
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = util.masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = util.masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2