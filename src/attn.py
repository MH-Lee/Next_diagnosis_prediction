import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.nn.functional as F


class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

    
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

    
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', dropout=0.1):
        super(TemporalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        weekday_size = 60; day_size = 32; month_size = 13; year_size=2500

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        self.weeks_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
        self.year_embed = Embed(year_size, d_model)
    
    def forward(self, x, time_feature):
        time_feature = time_feature.long()
        weeks_x = self.weeks_embed(time_feature[:,:,3])
        day_x = self.day_embed(time_feature[:,:,2])
        month_x = self.month_embed(time_feature[:,:,1])
        year_x = self.year_embed(time_feature[:,:,0])
        x = x + year_x + weeks_x + day_x + month_x
        temporal_embed = year_x + weeks_x + day_x + month_x
        return self.dropout(x), temporal_embed


class TimeEncoder(nn.Module):
    def __init__(self, ninp):
        super(TimeEncoder, self).__init__()
        self.weight_layer = torch.nn.Linear(ninp, ninp)

    def forward(self, time_encoder, final_queries, mask):
        selection_feature = F.relu(self.weight_layer(time_encoder))
        selection_feature = torch.sum(selection_feature * final_queries, 2, keepdim=True) 
        selection_feature = selection_feature.masked_fill_(mask, -1e8)
        return torch.softmax(selection_feature, 1)


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""
    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask is not None:
            attention = attention.masked_fill_(attn_mask.bool(), -1e8)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention
    
    
class MultiheadAttention(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiheadAttention, self).__init__()
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(query, key, value, scale, attn_mask)
        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)
        # final linear projection
        output = self.linear_final(context)
        # dropout
        output = self.dropout(output)
        # add residual and norm layer
        output = self.layer_norm(residual + output)
        return output, attention
    
    
class Decoder(nn.Module):
    def __init__(self, ninp, nhead, nlayers,  dropout=0.5):
        super(Decoder, self).__init__()
        self.emb_dropout = nn.Dropout(p=dropout)
        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(ninp, eps=1e-8)

        for _ in range(nlayers):
            new_attn_layernorm = nn.LayerNorm(ninp, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer =  MultiheadAttention(model_dim=ninp, num_heads=nhead, dropout=dropout)
            self.attention_layers.append(new_attn_layer)
            new_fwd_layernorm = nn.LayerNorm(ninp, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
            new_fwd_layer = nn.Sequential(
                nn.Linear(ninp, ninp*4),
                nn.ReLU(),
                nn.Linear(ninp*4, ninp)
            )
            self.forward_layers.append(new_fwd_layer)

    def forward(self, src, attention_mask):
        seqs = self.emb_dropout(src)
        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
        outputs = self.last_layernorm(seqs) # (PB, VL, H) 
        return outputs