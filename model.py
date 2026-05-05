import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

from helper import debug_parameters, get_settings_and_params, load_weight_into_gpt 

MODEL_NAME ="355M"
MODEL_DIR = os.path.join("models", MODEL_NAME)


if MODEL_NAME == "124M":
    VOCAB_SIZE = 50257
    CONTEXT_LEN = 1024
    NUM_HEADS = 12
    NUM_LAYER = 12
    EMB_DIM = 768
    DROP_RATE = 0.1
    QKV_BIAS = True
elif MODEL_NAME == "355M":
    VOCAB_SIZE = 50257
    CONTEXT_LEN = 1024
    NUM_HEADS = 20
    NUM_LAYER = 24
    EMB_DIM = 1280
    DROP_RATE = 0.1
    QKV_BIAS = True

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out,
                 CONTEXT_LEN, dropout,num_heads, qkv_bias = False):
        super().__init__()
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(CONTEXT_LEN, CONTEXT_LEN), diagonal=1))

        
    def forward(self, x):
        b, num_tokens,_ = x.size()
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores = attn_scores.masked_fill(mask_bool, torch.inf)
        attn_weights = torch.softmax(attn_scores/ keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
      
        return context_vec

    class LayerNorm(nn.Module):
       