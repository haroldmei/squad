import torch
from reformer_pytorch import Reformer

model = Reformer(
    dim = 512,
    depth = 12,
    max_seq_len = 8192,
    heads = 8,
    lsh_dropout = 0.1,
    causal = True
).cuda()

x = torch.randn(5, 8192, 512).cuda()
y = model(x) # (1, 8192, 512)

print(x.shape, y.shape)