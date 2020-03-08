"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""
import util

import layers
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer as transformer
import reformer as reformer
import qanet as qanet
#from qanet import Embedding, EncoderBlock, CQAttention, Initialized_Conv1d, Pointer

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
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.):
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

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs, cq_idxs):
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

# obsolete
class BiDAF_Transformer(nn.Module):
    """
    Use the similar framework as BiDAF but replace the attention mechanism from Transformer
    """
    
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.1):
        super(BiDAF_Transformer, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors, hidden_size=hidden_size, drop_prob=drop_prob)
        self.pemb = transformer.PositionalEncoding(hidden_size, drop_prob)
        self.enc = transformer.TransformerEncoder(hidden_size, N = 1)    # c = 4
        self.att = layers.BiDAFAttention(hidden_size=hidden_size, drop_prob=drop_prob)      # for test, from bidaf
        self.W = nn.Linear(4*hidden_size, hidden_size)
        self.mod = transformer.TransformerEncoder(hidden_size, N=3, c = 2)
        self.out = transformer.Transformer_Output(hidden_size=hidden_size, drop_prob=drop_prob)    # just want to run, don't think it will do anything.

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs, cq_idxs):
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
        mod = self.mod(self.W(att), c_mask)    
        out = self.out(att, mod, c_mask)  
        print(out[0].shape, out[1].shape)

        return out

        
# obsolete
class BiDAF_Transformer_Ex(nn.Module):
    """
    Use the similar framework as BiDAF but replace the attention mechanism from Transformer
    """
    
    def __init__(self, word_vectors, char_vectors, hidden_size, enc_layers=1, mod_layers = 3, kernel = 7, drop_prob=0.1):
        super(BiDAF_Transformer_Ex, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors, hidden_size=hidden_size, drop_prob=drop_prob)
        self.pemb = transformer.PositionalEncoding(hidden_size, drop_prob)
        self.cnn1 = transformer.TransformerEncoderLayerEx(hidden_size)
        self.enc = nn.modules.transformer.TransformerEncoder(
            nn.modules.transformer.TransformerEncoderLayer(hidden_size, 8, dropout=drop_prob), 
            enc_layers, 
            nn.modules.transformer.LayerNorm(hidden_size)
        )

        self.att = layers.BiDAFAttention(hidden_size=hidden_size, drop_prob=drop_prob)      # for test, from bidaf
        self.W = nn.Linear(4*hidden_size, hidden_size)
        self.cnn2 = transformer.TransformerEncoderLayerEx(hidden_size, c=2)
        self.mod = nn.modules.transformer.TransformerEncoder(
            nn.modules.transformer.TransformerEncoderLayer(hidden_size, 8, dropout=drop_prob), 
            mod_layers, 
            nn.modules.transformer.LayerNorm(hidden_size)
        )
        self.out = transformer.Transformer_OutputEx(hidden_size=hidden_size, mod_layers = mod_layers, drop_prob=drop_prob)    # just want to run, don't think it will do anything.

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs, cq_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs

        c_emb = self.emb(cw_idxs)
        q_emb = self.emb(qw_idxs)        

        c_enc = self.enc(self.cnn1(c_emb)) #, c_mask)   
        q_enc = self.enc(self.cnn1(q_emb)) #, q_mask)   

        att = self.att(c_enc, q_enc, c_mask, q_mask)    # 4 * hidden_size
        mod = self.mod(self.cnn2(self.W(att)))#, c_mask)    
        out = self.out(att, mod, c_mask)  

        return out

class BiDAF_QANet(nn.Module):
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.1):
        super().__init__()
        self.dropout = drop_prob

        self.char_emb = nn.Embedding.from_pretrained(torch.Tensor(char_vectors), freeze=False)
        self.word_emb = nn.Embedding.from_pretrained(torch.Tensor(word_vectors), freeze=True)
        self.emb = qanet.Embedding(word_vectors.size(1), char_vectors.size(1), hidden_size)
        
        self.emb_enc = qanet.EncoderBlock(conv_num=4, ch_num=hidden_size, k=7, dropout=drop_prob)
        self.cq_att = qanet.CQAttention(hidden_size)
        self.cq_resizer = qanet.Initialized_Conv1d(hidden_size * 4, hidden_size)
        self.model_enc_blks = nn.ModuleList([qanet.EncoderBlock(conv_num=2, ch_num=hidden_size, k=5, dropout=drop_prob) for _ in range(7)])
        self.out = qanet.Pointer(hidden_size)

    def forward(self, Cwid, Qwid, Ccid, Qcid, CQid):
        maskC = (torch.zeros_like(Cwid) != Cwid).float()
        maskQ = (torch.zeros_like(Qwid) != Qwid).float()
        
        Cw, Cc = self.word_emb(Cwid), self.char_emb(Ccid)
        Qw, Qc = self.word_emb(Qwid), self.char_emb(Qcid)

        C = self.emb(Cc, Cw)#.transpose(1,2)
        Q = self.emb(Qc, Qw)#.transpose(1,2)
        #print("3", C.shape, Q.shape)

        Ce = self.emb_enc(C, maskC, 1, 1)
        Qe = self.emb_enc(Q, maskQ, 1, 1)
        
        X = self.cq_att(Ce, Qe, maskC, maskQ)
        M0 = self.cq_resizer(X)
        M0 = F.dropout(M0, p=self.dropout, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, maskC, i*(2+2)+1, 7)
        M1 = M0
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, maskC, i*(2+2)+1, 7)
        M2 = M0
        M0 = F.dropout(M0, p=self.dropout, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, maskC, i*(2+2)+1, 7)
        M3 = M0
        p1, p2 = self.out(M1, M2, M3, maskC) # b, d, l
        
        return p1, p2


class BiDAF_Reformer(nn.Module):
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.1):
        super().__init__()
        self.dropout = drop_prob

        # use char embedding
        self.char_emb = nn.Embedding.from_pretrained(torch.Tensor(char_vectors), freeze=False)
        self.word_emb = nn.Embedding.from_pretrained(torch.Tensor(word_vectors), freeze=True)
        self.emb = qanet.Embedding(word_vectors.size(1), char_vectors.size(1), hidden_size)

        self.emb_enc = reformer.Reformer(
                                            dim = hidden_size,
                                            depth = 1,
                                            conv=4,
                                            kernel=7,
                                            bucket_size = 16,
                                            max_seq_len = 512,
                                            heads = 4,
                                            lsh_dropout = 0.1,
                                            causal = True,
                                            n_hashes = 16
                                        )

        self.cq_att = qanet.CQAttention(hidden_size)
        self.cq_resizer = qanet.Initialized_Conv1d(hidden_size * 4, hidden_size)
        self.model_enc_blks = nn.ModuleList([reformer.Reformer(
                                            dim = hidden_size,
                                            depth = 1,
                                            conv = 2,
                                            kernel=7,
                                            bucket_size = 16,
                                            max_seq_len = 512,
                                            heads = 8,
                                            lsh_dropout = 0.1,
                                            layer_dropout = 0.1,
                                            causal = True,
                                            n_hashes = 16
                                        ) for _ in range(2)])
        self.out = qanet.Pointer(hidden_size)

    def forward(self, Cwid, Qwid, Ccid, Qcid, CQid):
        maskC = (torch.zeros_like(Cwid) != Cwid).float()
        maskQ = (torch.zeros_like(Qwid) != Qwid).float()
        
        Cw, Cc = self.word_emb(Cwid), self.char_emb(Ccid)
        Qw, Qc = self.word_emb(Qwid), self.char_emb(Qcid)
        C, Q = self.emb(Cc, Cw).transpose(1,2), self.emb(Qc, Qw).transpose(1,2)

        Ce = self.emb_enc(C).transpose(1,2) #, maskC, 1, 1)
        Qe = self.emb_enc(Q).transpose(1,2) #, maskQ, 1, 1)
        
        X = self.cq_att(Ce, Qe, maskC, maskQ)
        M0 = self.cq_resizer(X)
        M0 = F.dropout(M0, p=self.dropout, training=self.training).transpose(1,2) # b, l, d
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0) 
        M1 = M0     # b, l, d
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0) 
        M2 = M0     # b, l, d
        M0 = F.dropout(M0, p=self.dropout, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0) 
        M3 = M0     # b, l, d
        
        p1, p2 = self.out(M1.transpose(1,2), M2.transpose(1,2), M3.transpose(1,2), maskC)   # # b, d, l
        
        #print(M1.shape,M2.shape,M3.shape, p1.sum(dim=1))
        return p1, p2
