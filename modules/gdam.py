import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from R2Gen.modules.visual_extractor import VisualExtractor
from R2Gen.modules.base_cmn import BaseCMN
from torch.nn import Parameter


# CADA
class ContextAwareDynamicAlignment(nn.Module):
    def __init__(self, key_dim, value_dim, lfp=512):
        super().__init__()
        self.key_proj = nn.Linear(key_dim, lfp, bias=False)
        self.val_proj = nn.Linear(value_dim, lfp, bias=False)
        self.attn_proj = nn.Linear(lfp, 1, bias=False)

    def forward(self, key, value):
        joint_repr = self.key_proj(key).unsqueeze(1) + self.val_proj(value)
        attn_scores = self.attn_proj(torch.tanh(joint_repr)).squeeze(2)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = (value * attn_weights.unsqueeze(2)).sum(dim=1)
        return context, attn_weights


    #TSA
class TemporalSaliencyGatedAttention(nn.Module): #temporal salience attention 3.2.5 section
    def __init__(self, key_dim, value_dim, cad=768):
        super().__init__()
        self.key_proj = nn.Linear(key_dim, cad, bias=False)
        self.val_proj = nn.Linear(value_dim, cad, bias=False)
        self.attn_proj = nn.Linear(cad, 1, bias=False)

    def forward(self, key, value):
        key_rep = self.key_proj(key).unsqueeze(1)
        val_rep = self.val_proj(value)
        interaction = torch.tanh(key_rep + val_rep)
        attn_scores = self.attn_proj(interaction).squeeze(2)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = (value * attn_weights.unsqueeze(2)).sum(dim=1)
        return context, attn_weights


      
        # input_dim = 64
        # num_classes = 10
        # key_dim = 128
        # value_dim = 256
        # iu_xray_identifiers = ["CXR2_IM-0652-T1", "CXR2_IM-0652-T2", "CXR2_IM-0652-T3"]
        # feature_maps = torch.randn(4, input_dim, 16, 16)
        # key_tensor = torch.randn(4, key_dim)
        # value_tensor = torch.randn(4, value_dim)
        #
        # module = IUXrayAttentionModule(input_dim, num_classes, key_dim, value_dim)
        # output = module(iu_xray_identifiers, feature_maps, key_tensor, value_tensor)
        #
        # for identifier, result in output.items():
        #     print(f"Results for {identifier}:")
        #     for key, value in result.items():
        #         print(f"  {key}: {value.shape}")

