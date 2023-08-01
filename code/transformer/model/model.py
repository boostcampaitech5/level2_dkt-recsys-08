import copy
from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class DKTModelBase(BaseModel):
    def __init__(self, args):
        super().__init__()

        self.cat_cols = args.cat_cols
        self.num_cols = args.num_cols

        self.embddings = nn.ModuleList(
            [nn.Embedding(dim + 1, args.emb_dim) for dim in args.dim_cats]
        )

        self.cat_proj = nn.Linear(
            args.emb_dim * len(args.cat_cols), args.hidden_dim // 2
        )
        self.num_proj = nn.Linear(len(args.num_cols), args.hidden_dim // 2)

        self.cat_norm = nn.LayerNorm(args.hidden_dim // 2).to(args.device)
        self.num_norm = nn.LayerNorm(args.hidden_dim // 2).to(args.device)

        # output layer
        self.output = nn.Linear(args.hidden_dim, 1)

    def forward(self, data):
        # Embedding
        embed_cat = torch.cat(
            [
                self.embddings[i](data[self.cat_cols[i]].int())
                for i in range(len(self.cat_cols))
            ],
            dim=2,
        )
        embed_num = torch.cat([data[col].unsqueeze(2) for col in self.num_cols], dim=2)

        embed_cat = self.cat_norm(self.cat_proj(embed_cat))
        embed_num = self.num_norm(self.num_proj(embed_num))

        embed = torch.cat([embed_cat, embed_num], dim=2)
        return embed, data["mask"]


class TransformerEncoder(DKTModelBase):
    def __init__(self, args):
        super(TransformerEncoder, self).__init__(args)
        self.layers = nn.ModuleList(
            [copy.deepcopy(EncoderBlock(args)) for _ in range(args.n_layers)]
        )

    def make_mask(self, source, target, pad_idx=0):
        source_seq_len, target_seq_len = source.size(1), target.size(1)

        target_mask = target.ne(pad_idx).unsqueeze(1).unsqueeze(2)
        target_mask = target_mask.repeat(1, 1, source_seq_len, 1)

        source_mask = source.ne(pad_idx).unsqueeze(1).unsqueeze(3)
        source_mask = source_mask.repeat(1, 1, 1, target_seq_len)

        mask = source_mask & target_mask
        mask.requires_grad = False

        return mask

    def forward(self, data):
        out, mask = super().forward(data)
        mask = self.make_mask(mask, mask)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.output(out).view(out.size(0), -1)
        return out


class EncoderBlock(nn.Module):
    def __init__(self, args):
        super(EncoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(args)
        self.feed_forward = FeedForward(args)
        self.adds = [Add_Norm(args) for _ in range(2)]

    def forward(self, x, mask=None):
        out = x
        out = self.adds[0](out, lambda out: self.self_attention(out, out, out, mask))
        out = self.adds[1](out, self.feed_forward)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()
        self.d_model = args.hidden_dim * args.n_heads
        self.h = args.n_heads
        self.q_fc = nn.Linear(args.hidden_dim, self.d_model)
        self.k_fc = nn.Linear(args.hidden_dim, self.d_model)
        self.v_fc = nn.Linear(args.hidden_dim, self.d_model)
        self.out_fc = nn.Linear(self.d_model, args.hidden_dim)

    def forward(self, Q, K, V, mask=None):
        n_batch = Q.size(0)

        def transform(x, fc):
            out = fc(x)
            out = out.view(n_batch, -1, self.h, self.d_model // self.h)
            out = out.transpose(1, 2)
            return out

        Q = transform(Q, self.q_fc)
        K = transform(K, self.k_fc)
        V = transform(V, self.v_fc)

        def calculate_attention(Q, K, V, mask):
            d_k = K.shape[-1]
            attention_score = torch.matmul(Q, K.transpose(-2, -1))
            attention_score = attention_score / sqrt(d_k)
            if mask is not None:
                attention_score = attention_score.masked_fill(mask == 0, -1e12)
            attention_score = torch.softmax(attention_score, dim=-1)
            out = torch.matmul(attention_score, V)
            return out

        out = calculate_attention(Q, K, V, mask)
        out = out.transpose(1, 2)
        out = out.contiguous().view(n_batch, -1, self.d_model)
        out = self.out_fc(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, args):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(args.hidden_dim, args.mlp_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(args.mlp_dim, args.hidden_dim)

    def forward(self, x):
        out = x
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out


class Add_Norm(nn.Module):
    def __init__(self, args):
        super(Add_Norm, self).__init__()
        self.norm = nn.LayerNorm(args.hidden_dim).to(args.device)
        self.drop_out = nn.Dropout(args.drop_out)

    def forward(self, x, layer):
        out = x
        out = layer(out)
        out = self.norm(self.drop_out(out) + x)
        return out
