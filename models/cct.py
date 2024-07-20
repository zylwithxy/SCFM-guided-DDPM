from dataclasses import dataclass
import torch.nn as nn
import torch
from typing import List
from torch.nn.modules.utils import _pair
import copy
import math
import numpy as np

@dataclass
class CCTConfig():
    image_size: int = 64
    patchSizes: List[int] = None # patch size for each feature map
    fmapSizes: List[int] = None # feature map size for each feature map
    channelDims: List[int] = None # Dimension for each feature map
    query_index: List[int] = 0 # The indexes of queries in all feature map
    embedding_droprate: float = 0.1
    encoder_layer: int = 4 # The number of encoder layer
    encoder_attn_visual: bool = False # Visualize attention weights in encoder
    block_vit_expand_ratio: int = 4 # For the expand ratio in mlp of vit block
    attn_head_num: int = 4 # The head num for multihead attention block
    attention_dropout_rate: float = 0.1
    dropout_rate: float = 0.0
    def make_model(self):
        return ChannelTransformer(self)


class ChannelTransformer(nn.Module):
    def __init__(self, conf: CCTConfig):
        super().__init__()

        self.conf = conf
        
        assert len(conf.patchSizes) == len(conf.fmapSizes) == len(conf.channelDims)
        for idx, patchsize in enumerate(conf.patchSizes):
            setattr(self, f'patchSize_{idx+1}', patchsize)

        for idx, query_idx in enumerate(conf.query_index):
            temp = Channel_Embeddings(conf.embedding_droprate, conf.patchSizes[query_idx], img_size=conf.fmapSizes[query_idx], in_channels=conf.channelDims[query_idx])
            setattr(self, f'embeddings_{idx+1}', temp)
        
        self.encoder = Encoder(conf, conf.encoder_attn_visual, conf.channelDims)

        for idx, query_idx in enumerate(conf.query_index):
            reconstruct = Reconstruct(conf.channelDims[query_idx], conf.channelDims[query_idx], kernel_size= 1,scale_factor= (conf.patchSizes[query_idx], conf.patchSizes[query_idx]))
            setattr(self, f'reconstruct_{idx+1}', reconstruct)

    def forward(self, encs: List[torch.Tensor]):

        emb_encs = [None for _ in range(len(encs))]

        # for idx, enc in enumerate(encs):
        for idx, q_idx in enumerate(self.conf.query_index):
            embeddings = getattr(self, f'embeddings_{idx+1}')
            emb_encs[q_idx] = embeddings(encs[q_idx])

        emb_norms, attn_weights = self.encoder(*emb_encs)  # (B, n_patch, hidden)

        outputs = [None for _ in range(len(encs))]
        for idx, q_idx in enumerate(self.conf.query_index):
            reconstruct = getattr(self, f'reconstruct_{idx+1}')
            outputs[q_idx] = reconstruct(emb_norms[q_idx]) + encs[q_idx]

        return outputs, attn_weights


class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        if x is None:
            return None

        B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = nn.Upsample(scale_factor=self.scale_factor)(x)

        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class Channel_Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, droprate, patchsize, img_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patchsize)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = nn.Dropout(droprate)

    def forward(self, x):
        if x is None:
            return None
        
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Encoder(nn.Module):
    def __init__(self, conf: CCTConfig, vis, channel_nums):
        super(Encoder, self).__init__()
        self.conf = conf
        self.vis = vis
        self.layer = nn.ModuleList()

        # for idx, ch in enumerate(channel_nums):
        for idx, q_idx in enumerate(conf.query_index):
            setattr(self, f'encoder_norm{idx+1}', nn.LayerNorm(channel_nums[q_idx], eps=1e-6))

        for _ in range(conf.encoder_layer):
            layer = Block_ViT(conf, vis, channel_nums)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, *emb_encs):

        attn_weights = []

        for layer_block in self.layer:
            emb_encs, weights = layer_block(*emb_encs) 
            if self.vis:
                attn_weights.append(weights)
        
        emb_norms = [None for _ in range(len(emb_encs))]
        # for idx, enc in enumerate(emb_encs):
        for idx, q_idx in enumerate(self.conf.query_index):
            encoder_norm_layer = getattr(self, f'encoder_norm{idx+1}')
            emb_norms[q_idx] = encoder_norm_layer(emb_encs[q_idx])
        
        return emb_norms, attn_weights


class Block_ViT(nn.Module):
    def __init__(self, conf: CCTConfig, vis, channel_nums):
        super(Block_ViT, self).__init__()
        self.conf = conf
        expand_ratio = conf.block_vit_expand_ratio

        # for idx, ch in enumerate(channel_nums):
        for idx, q_idx in enumerate(conf.query_index):
            setattr(self, f'attn_norm{idx+1}', nn.LayerNorm(channel_nums[q_idx], eps=1e-6))

        self.attn_norm =  nn.LayerNorm(sum([channel_nums[q_idx] for q_idx in conf.query_index]),eps=1e-6)
        self.channel_attn = Attention_org(conf, vis, channel_nums)

        # for idx, ch in enumerate(channel_nums):
        for idx, q_idx in enumerate(conf.query_index):
            setattr(self, f'ffn_norm{idx+1}', nn.LayerNorm(channel_nums[q_idx], eps=1e-6))

        # for idx, ch in enumerate(channel_nums):
        for idx, q_idx in enumerate(conf.query_index):
            setattr(self, f'ffn{idx+1}', Mlp(conf, channel_nums[q_idx], channel_nums[q_idx]*expand_ratio))
    
    def forward(self, *emb_encs):
        embcat = []

        for q_idx in self.conf.query_index:
            embcat.append(emb_encs[q_idx])

        emb_all = torch.cat(embcat,dim=2)
        
        cx_list = [None for _ in range(len(emb_encs))]
        for idx, q_idx in enumerate(self.conf.query_index):
            attn_norm_layer = getattr(self, f'attn_norm{idx+1}')
            cx_list[q_idx] = attn_norm_layer(emb_encs[q_idx])

        emb_all = self.attn_norm(emb_all)
        cx_list, weights = self.channel_attn(cx_list, emb_all)

        for q_idx in self.conf.query_index:
            cx_list[q_idx] = cx_list[q_idx] + emb_encs[q_idx]

        outputs = [None for _ in range(len(emb_encs))]
        for idx, q_idx in enumerate(self.conf.query_index):
            norm_layer = getattr(self, f'ffn_norm{idx+1}')
            mlp = getattr(self, f'ffn{idx+1}')
            outputs[q_idx] = mlp(norm_layer(cx_list[q_idx]))

        for idx, q_idx in enumerate(self.conf.query_index):
            outputs[q_idx] = outputs[q_idx] + cx_list[q_idx]
       
        return outputs, weights


class Attention_org(nn.Module):
    def __init__(self, conf: CCTConfig, vis, channel_nums):
        super(Attention_org, self).__init__()
        self.conf = conf
        self.vis = vis
        self.KV_size = sum([channel_nums[q_idx] for q_idx in conf.query_index])
        self.channel_nums = channel_nums
        self.num_attention_heads = conf.attn_head_num

        for idx in range(len(conf.query_index)):
            setattr(self, f'query{idx+1}', nn.ModuleList())

        self.key = nn.ModuleList()
        self.value = nn.ModuleList()
        
        queries = []
        for q_idx in conf.query_index:
            queries.append(nn.Linear(channel_nums[q_idx], channel_nums[q_idx], bias=False))

        for _ in range(self.num_attention_heads):
            key = nn.Linear(self.KV_size,  self.KV_size, bias=False)
            value = nn.Linear(self.KV_size,  self.KV_size, bias=False)
            
            for idx in range(len(conf.query_index)):
                getattr(self, f'query{idx+1}').append(copy.deepcopy(queries[idx]))
        
            self.key.append(copy.deepcopy(key))
            self.value.append(copy.deepcopy(value))

        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = nn.Softmax(dim=3)

        for idx, q_idx in enumerate(conf.query_index):
            setattr(self, f'out{idx+1}',nn.Linear(channel_nums[q_idx], channel_nums[q_idx], bias=False))
    
        self.attn_dropout = nn.Dropout(conf.attention_dropout_rate)
        self.proj_dropout = nn.Dropout(conf.attention_dropout_rate)

    def forward(self, cx_list, emb_all):
        multi_head_Q_list = [[] for _ in range(len(self.conf.query_index))]
        multi_head_K_list = []
        multi_head_V_list = []

        for idx, q_idx in enumerate(self.conf.query_index):
            for query in getattr(self, f'query{idx+1}'):
                multi_head_Q_list[idx].append(query(cx_list[q_idx]))

        for key in self.key:
            K = key(emb_all)
            multi_head_K_list.append(K)
        for value in self.value:
            V = value(emb_all)
            multi_head_V_list.append(V)
        
        multi_head_K = torch.stack(multi_head_K_list, dim=1)
        multi_head_V = torch.stack(multi_head_V_list, dim=1)
        attention_probs = []

        for idx in range(len(multi_head_Q_list)):
            multi_head_Q_list[idx] = torch.stack(multi_head_Q_list[idx], dim=1)
            multi_head_Q_list[idx] = multi_head_Q_list[idx].transpose(-1, -2)
            attention_probs.append(self.softmax(self.psi(torch.matmul(multi_head_Q_list[idx], multi_head_K) / math.sqrt(self.KV_size))))
    
        weights =  [attn_prob.mean(1) for attn_prob in attention_probs] if self.vis else None

        attention_probs = [self.attn_dropout(attn_prob) for attn_prob in attention_probs]
       
        multi_head_V = multi_head_V.transpose(-1, -2)
        context_layers = [torch.matmul(attn_prob, multi_head_V) for attn_prob in attention_probs]
        context_layers = [context.permute(0, 3, 2, 1).contiguous().mean(dim= 3) for context in context_layers]
        
        outputs = []
        for idx in range(len(self.conf.query_index)):
            out_layer = getattr(self, f'out{idx+1}')
            outputs.append(self.proj_dropout(out_layer(context_layers[idx])))

        for idx, q_idx in enumerate(self.conf.query_index):
             cx_list[q_idx] = outputs[idx]
            
        return cx_list, weights
        # return outputs, weights


class Mlp(nn.Module):
    def __init__(self, conf: CCTConfig, in_channel, mlp_channel):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_channel, mlp_channel)
        self.fc2 = nn.Linear(mlp_channel, in_channel)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(conf.dropout_rate)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x