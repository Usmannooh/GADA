import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from R2Gen.modules.visual_extractor import VisualExtractor
from R2Gen.modules.base_cmn import BaseCMN
from torch.nn import Parameter


class DynamicGraphAttension(nn.Module):
    def __init__(self, key_dim, value_dim, LFP=512):
        super(DynamicGraphAttension, self).__init__()
        self.key_lfp = nn.Linear(key_dim, LFP, bias=False)
        self.value_lfp = nn.Linear(value_dim, LFP, bias=False)
        self.attention_lfp = nn.Linear(LFP, 1, bias=False)

    def forward(self, key_tensor, value_tensor):
        interaction = self.key_lfp(key_tensor).unsqueeze(1) + self.value_lfp(value_tensor)
        attention_scores = self.attention_lfp(torch.tanh(interaction)).squeeze(2)
        attention_weights = torch.softmax(attention_scores, dim=1)
        attention_context = (value_tensor * attention_weights.unsqueeze(2)).sum(dim=1)

        return attention_context, attention_weights



class KeyEventAttention(nn.Module):
    def __init__(self, key_dim, value_dim, CAD=1024):
        super(KeyEventAttention, self).__init__()
        self.key_cad = nn.Linear(key_dim, CAD, bias=False)
        self.value_cad = nn.Linear(value_dim, CAD, bias=False)
        self.attention_cad = nn.Linear(CAD, 1, bias=False)

    def forward(self, key_tensor, value_tensor):

        key_cad = self.key_cad(key_tensor).unsqueeze(1)
        value_cad = self.value_cad(value_tensor)
        interaction = key_cad + value_cad
        attention_scores = self.attention_cad(torch.tanh(interaction)).squeeze(2)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context = (value_tensor * attention_weights.unsqueeze(2)).sum(dim=1)

        return context, attention_weights

class HybridTransformerPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(HybridTransformerPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, lengths):
        mask = torch.arange(x.size(1))[None, :] < lengths[:, None]
        x = x + self.pe[:x.size(0), :] * mask.unsqueeze(-1)
        return x


