"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""
import util

import layers
import torch
import torch.nn as nn


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, cq_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


class BiDAF_Transformer(nn.Module):
    """
    Use the similar framework as BiDAF but replace the attention mechanism from Transformer
    """
    
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF_Transformer, self).__init__()

        self.device, _ = util.get_available_devices()

        self.emb = layers.Embedding(word_vectors=word_vectors, hidden_size=hidden_size, drop_prob=drop_prob)

        self.pemb = layers.PositionalEncoding(hidden_size, drop_prob)

        self.enc = layers.TransformerEncoder(hidden_size, N = 1)    # b = 1, c = 4

        self.att = layers.BiDAFAttention(hidden_size=hidden_size, drop_prob=drop_prob)      # for test, from bidaf

        self.W = nn.Linear(4*hidden_size, hidden_size)

        self.m0 = layers.TransformerEncoder(hidden_size, b = 3, c = 2)

        self.out = layers.Transformer_Output(hidden_size=hidden_size, drop_prob=drop_prob)    # just want to run, don't think it will do anything.

    def forward(self, cw_idxs, qw_idxs, cq_idxs):
        """
        this is hard.
        """
        #cq_mask = torch.zeros_like(cq_idxs) != cq_idxs
        #zeros = torch.zeros(cq_idxs.shape, dtype=torch.int64)
        #zeros = zeros.to(self.device)
        #zeros[:,:cw_idxs.shape[1]] = cw_idxs
        #c_mask_ = torch.zeros_like(zeros) != zeros

        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs

        c_emb = self.emb(cw_idxs)        
        q_emb = self.emb(qw_idxs)        

        c_enc = self.enc(c_emb, c_mask)   
        q_enc = self.enc(q_emb, q_mask)   

        att = self.att(c_enc, q_enc, c_mask, q_mask)    # 4 * hidden_size

        mod = self.W(att)
        mod0 = self.m0(mod, c_mask)    
        mod1 = self.m0(mod0, c_mask)    

        out = self.out(att, mod0, mod1, c_mask)  

        return out


class BiDAF_Reformer(nn.Module):
    """
    Use the similar framework as BiDAF but replace tghe attention mechanism from Reformer
    """

    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF_Reformer, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, cq_idxs):
        """
        this is hard.
        """
        print("BiDAF_Reformer.forward")