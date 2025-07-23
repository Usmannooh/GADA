import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from R2Gen.modules.visual_extractor import VisualExtractor
from R2Gen.modules.base_cmn import BaseCMN
from torch.nn import Parameter
from R2Gen.modules.gdam import DynamicGraphAttension,KeyEventAttention

class RGCN(nn.Module):
    def __init__(self, input_dim, output_dim, gcn_layers=3, activation=nn.GELU()):
        super(RGCN, self).__init__()

        if gcn_layers is not None:
            self.gcn_layers = nn.ModuleList()
            self.gcn_layers.append(GraphConvolution(input_dim, output_dim))
            for _ in range(1, gcn_layers):
                self.gcn_layers.append(GraphConvolution(output_dim, output_dim))
        else:
            self.gcn_layers = None
        self.activation = activation

    def forward(self, states, forward_adj, backward_adj):

        if forward_adj is None or backward_adj is None:
            raise ValueError("Both forward and backward adjacency matrices must be provided.")

        if self.gcn_layers is not None:
            states = states.permute(0, 2, 1)
            for i, layer in enumerate(self.gcn_layers):
                forward_states = self.activation(layer(states, forward_adj))
                backward_states = self.activation(layer(states, backward_adj))
                states = (forward_states + backward_states) / 2
            states = states.permute(0, 2, 1)


        return states

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_matrix = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.batch_norm = nn.BatchNorm1d(out_features)
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self._initialize_parameters()

    def _initialize_parameters(self):

        nn.init.xavier_normal_(self.weight_matrix, gain=0.02)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, input_features, adjacency_matrix):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = self.weight_matrix.device
        input_features = input_features.to(device)
        adjacency_matrix = adjacency_matrix.to(device)
        input_features = input_features.permute(0, 2, 1)
        support_matrix = torch.bmm(input_features,
                                   self.weight_matrix.unsqueeze(0).expand(input_features.size(0), -1, -1))
        convolved_features = torch.bmm(adjacency_matrix, support_matrix)
        convolved_features = self.batch_norm(convolved_features)

        if self.bias is not None:
            convolved_features += self.bias

        return convolved_features.permute(0, 2, 1)



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

class Sman(nn.Module):
    def __init__(self, args, tokenizer, num_classes, forward_adj, backward_adj, feature_dim=2048, embed_size=256,
                 hidden_size=612):
        super(Sman, self).__init__()

        self.args = args
        self.num_classes = num_classes
        self.tokenizer = tokenizer
        self.feature_dim = feature_dim
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.forward_adj = forward_adj
        self.backward_adj = backward_adj
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = BaseCMN(args, tokenizer)
        self.graph_cn = RGCN(feature_dim, feature_dim )
        self.attention = DynamicGraphAttension(hidden_size, feature_dim)
        self.key_event_attention = KeyEventAttention(key_dim=hidden_size, value_dim=feature_dim)



        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad == True, self.parameters())
        total_params = sum(np.prod(p.size()) for p in model_parameters)
        return f'{self.__class__.__name__}\nTrainable parameters: {total_params}'

    def init_hidden(self, batch_size):
        device = torch.device('cuda')
        hidden_state = torch.zeros(self.args.num_layers, batch_size, self.hidden_size, device=device)
        cell_state = torch.zeros(self.args.num_layers, batch_size, self.hidden_size, device=device)
        return hidden_state, cell_state

    def forward(self, img_features, enc_features, captions, **kwargs):
        batch_size = img_features.size(0)


        img_features = self.positional_encoding(img_features)
        context_features, alpha = self.attention(enc_features, img_features)
        context_features = context_features.view(batch_size, self.feature_dim, 1, 1)
        visual_features = self.visual_extractor(img_features)
        visual_features = self.graph_cn(visual_features, forward_adj=self.normalized_forward_adj,
                                        backward_adj=self.normalized_backward_adj)
        visual_features = self.class_attention(visual_features)

        # Flatten visual features for captioning
        flattened_visual = visual_features.view(batch_size, -1)
        self.hidden = (self.init_sent_h(flattened_visual), self.init_sent_c(flattened_visual))
        output_captions = self.captioning(visual_features, captions)  # Image Captioning

        return output_captions, alpha

    def forward_iu_xray(self, images, captions=None, mode='train', update_opts={}):
        att_features = []
        func_features = []

        for i in range(2):
            att_feat, func_feat = self.visual_extractor(images[:, i])
            att_features.append(att_feat)
            func_features.append(func_feat)

        func_feature = torch.cat(func_features, dim=1)
        forward_adj = self.normalized_forward_adj.repeat(6, 1, 1)
        backward_adj = self.normalized_backward_adj.repeat(6, 1, 1)
        global_features = [feat.mean(dim=(2, 3)) for feat in att_features]
        att_features = [self.class_attention(feat, self.num_classes) for feat in att_features]

        for idx in range(2):
            att_features[idx] = torch.cat((global_features[idx].unsqueeze(1), att_features[idx]), dim=1)
            att_features[idx] = self.linear_trans_lyr_2(att_features[idx].transpose(1, 2)).transpose(1, 2)
        att_feature_combined = torch.cat(att_features, dim=1)
        att_feature_combined = self.linear_trans_lyr(att_feature_combined.transpose(1, 2)).transpose(1, 2)

        if mode == 'train':
            return self.encoder_decoder(func_feature, att_feature_combined, captions, mode='forward')
        elif mode == 'sample':
            return self.encoder_decoder(func_feature, att_feature_combined, mode='sample', update_opts=update_opts)
        else:
            raise ValueError("Invalid mode provided.")

    def forward_mimic_cxr(self, images, captions=None, mode='train', update_opts={}):
        att_features, func_feature = self.visual_extractor(images)
        if mode == 'train':
            return self.encoder_decoder(func_feature, att_features, captions, mode='forward')
        elif mode == 'sample':
            return self.encoder_decoder(func_feature, att_features, mode='sample', update_opts=update_opts)
        else:
            raise ValueError("Invalid mode provided.")
