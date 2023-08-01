import copy
from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torch_geometric.nn.models import LightGCN
from model.position_encoding import PositionalEncoding
import numpy as np

class DKTModelBase(BaseModel):
    def __init__(self, hidden_dim, n_layers, n_tests, n_questions, n_tags):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_tests = n_tests
        self.n_questions = n_questions
        self.n_tags = n_tags

        # Embeddings
        partial_dimension = hidden_dim // 4
        self.embedding_interaction = nn.Embedding(3, partial_dimension)
        self.embedding_test = nn.Embedding(n_tests + 1, partial_dimension)
        self.embedding_question = nn.Embedding(n_questions + 1, partial_dimension)
        self.embedding_tag = nn.Embedding(n_tags + 1, partial_dimension)

        # output layer
        self.output = nn.Linear(hidden_dim, 1)
    
    def forward(self, test, question, tag, correct, mask, interaction):
        # Embedding
        embed_interaction = self.embedding_interaction(interaction.int())
        embed_test = self.embedding_test(test.int())
        embed_question = self.embedding_question(question.int())
        embed_tag = self.embedding_tag(tag.int())
        embed = torch.cat([embed_interaction, embed_test, embed_question, embed_tag], dim=2)
        return embed, mask

class GTN(DKTModelBase):
    def __init__(self, args):
        super(GTN, self).__init__(args.hidden_dim, args.n_layers, args.n_tests, args.n_questions, args.n_tags)
        self.step_wise = TransformerEncoder(args)
        self.channel_wise = TransformerEncoder_Position(args)
        self.pred = nn.Sequential(
            nn.Linear(self.hidden_dim*2, self.hidden_dim*2),
            nn.Dropout(p=args.drop_out),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, 1)
        )

    def forward(self, test, question, tag, correct, mask, interaction):
        out, mask = super().forward(test, question, tag, correct, mask, interaction)

        step_out = self.step_wise(out, mask)
        channel_out = self.channel_wise(out, mask)

        out = torch.cat([step_out, channel_out], dim=-1)
        out = self.pred(out).view(test.size(0), -1)
        return out
    
class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(EncoderBlock(args)) for _ in range(args.n_layers)])

    def make_mask(self, source, target, pad_idx=0):
        source_seq_len, target_seq_len = source.size(1), target.size(1)

        target_mask = target.ne(pad_idx).unsqueeze(1).unsqueeze(2)
        target_mask = target_mask.repeat(1, 1, source_seq_len, 1) 

        source_mask = source.ne(pad_idx).unsqueeze(1).unsqueeze(3)
        source_mask = source_mask.repeat(1, 1, 1, target_seq_len) 

        mask = source_mask & target_mask
        mask.requires_grad = False
        
        return mask
    
    def forward(self, x, mask = None):
        out = x
        mask = self.make_mask(mask, mask)
        for layer in self.layers:
            out = layer(out, mask)
        return out

class TransformerEncoder_Position(nn.Module):
    def __init__(self, args):
        super(TransformerEncoder_Position, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(EncoderBlock(args)) for _ in range(args.n_layers)])
        self.positional_emb = PositionalEncoding(args.hidden_dim, args.max_seq_len, device='cuda')

    def make_mask(self, target, pad_idx=0):
        source = torch.cuda.BoolTensor(target == 1).unsqueeze(1)
        target_seq_len = target.size(1)
        
        target_mask = torch.from_numpy((1-np.triu(np.ones((1, target_seq_len, target_seq_len)), k=1)).astype('bool')).to(torch.device('cuda'))
        target = (source & target_mask)
        target.requires_grad = False
        
        return target.unsqueeze(1)
    
    def forward(self, x, mask = None):
        out = x
        out = out + self.positional_emb(out)
        mask = self.make_mask(mask)
        for layer in self.layers:
            out = layer(out, mask)
        
        return out
    
class EncoderBlock(nn.Module):
    def __init__(self, args):
        super(EncoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(args)
        self.feed_forward = FeedForward(args)
        self.adds = [Add_Norm(args) for _ in range(2)]
        self.position = False
        if args.use_each_position:
            self.position = True
            self.positional_emb = PositionalEncoding(args.hidden_dim, args.max_seq_len, device='cuda')

    def forward(self, x, mask = None):
        out = x
        if self.position:
            out += self.positional_emb(x)
        out = self.adds[0](out, lambda out : self.self_attention(out, out, out, mask))
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

    def forward(self, Q, K, V, mask = None):
        n_batch = Q.size(0)

        def transform(x, fc):
            out = fc(x)
            out = out.view(n_batch, -1, self.h, self.d_model // self.h)
            out = out.transpose(1,2)
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
            attention_score = torch.softmax(attention_score, dim = -1)
            out = torch.matmul(attention_score, V)
            return out

        out = calculate_attention(Q, K, V, mask)
        out = out.transpose(1,2)
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
        self.norm = nn.LayerNorm(args.hidden_dim).cuda()
        self.drop_out = nn.Dropout(args.drop_out)

    def forward(self, x, layer):
        out = x
        out = layer(out)
        out = self.norm(self.drop_out(out) + x)
        return out