import math
from inspect import isfunction
from functools import partial

#matplotlib inline
import matplotlib.pyplot as plt
#from tqdm.auto import tqdm
#from einops import rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F


###Want to add an attention layer in here, Phil Wang does this, and it goes before the residual
###connection in the UNet. Will likely update this later
class CrossAttention(nn.Module):
    def __init__(self, intermediate_dim, num_heads, head_dim, attn_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim **-0.5

        attn_dim = num_heads * head_dim
        self.q = nn.Linear(intermediate_dim, attn_dim)
        self.k = nn.Linear(intermediate_dim, attn_dim)
        self.v = nn.Linear(intermediate_dim, attn_dim)

        self.out = nn.Sequential(nn.Linear(attn_dim, intermediate_dim))
    
    def forward(self, x, embeds):
        if embeds is None:
            embeds = x
        
        batch_size, len_q, len_k, len_v= q.size(0), q.size(1), k.size(1), v.size(1)

        q = self.q(x).view(batch_size, len_q, self.num_heads, self.head_dim)
        k = self.k(embeds).view(batch_size, len_k, self.num_heads, self.head_dim)
        v = self.v(embeds).view(batch_size, len_v, self.num_heads, self.head_dim)

        attention_values = torch.matmul(q / self.scale, k.transpose(2, 3))
        attention_values = self.dropout(F.softmax(attention_values, dim=-1))
        q = torch.matmul(attention_values, v)

        q = q.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        return self.out(q)


##Use basically same thing as Transformers, take in (batch_size, 1) --> (batch_size, embed_dim)
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, time):
        device = time.device
        half_dim = self.embed_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
