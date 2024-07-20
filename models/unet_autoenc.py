from enum import Enum

import torch
from torch import Tensor
from torch.nn.functional import silu

from .latentnet import *
from .unet import *
from .choices import *
from models.flownet import FlowNetConfig
from models.target_net import PoseTargetConfig
from models.networks.resample2d_package.resample2d import Resample2d
from models.cct import CCTConfig
from typing import List

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob


@dataclass
class BeatGANsAutoencConfig(BeatGANsUNetConfig):
    # number of style channels
    enc_out_channels: int = 512
    enc_attn_resolutions: Tuple[int] = None
    enc_pool: str = 'depthconv'
    enc_num_res_block: int = 2
    enc_channel_mult: Tuple[int] = None
    enc_grad_checkpoint: bool = False
    latent_net_conf: MLPSkipNetConfig = None
    use_warp: bool = False
    flownet_attnlayer: List[int] = None
    use_warp_separate: bool = False # generate the same resolution warp separately, for example (32, 32)
    use_attention_qkv: bool = True # if use qkv attention in Attention block
    ablation_pose_target_wo_target_enc: bool = False # True means using part of cond such as (b, c, 32, 32) and downsample these features to (16, 16) when use_warp == False, while generating flow_lists and masks
    enable_architecture_remove: bool = False # for removing input_blocks and output_blocks
    enable_source_enc_remove: bool = False # for removing source encoder blocks
    enable_archi_inres_remove: bool = False
    enable_source_enc_inres_remove: bool = False
    use_cct: bool = False # whether to use the CCT module.
    cct_pos: str = 'input_blocks' # cct_pos must be in ['input_blocks', 'source_encoder']
    use_flowdisp: bool = False # if use the displacement map and confidence map for optical flow

    def make_model(self):
        return BeatGANsAutoencModel(self)


class BeatGANsAutoencModel(BeatGANsUNetModel):
    def __init__(self, conf: BeatGANsAutoencConfig):
        super().__init__(conf)
        self.conf = conf

        # having only time, cond
        self.time_embed = TimeStyleSeperateEmbed(
            time_channels=conf.model_channels,
            time_out_channels=conf.embed_channels,
        )

        conf.in_channels = 3
        self.encoder = BeatGANsEncoder(conf)

        if conf.latent_net_conf is not None:
            self.latent_net = conf.latent_net_conf.make_model()
        
        self.flow_net = FlowNetConfig(attn_layer=conf.flownet_attnlayer, use_warp_separate= conf.use_warp_separate).make_model()
        if conf.use_cct:
            assert conf.cct_pos in ['input_blocks', 'source_encoder', 'source_encoder_inres_not']
            
            if conf.cct_pos == 'input_blocks':
                self.key_candidates_num = 3
                patchsizes = [16] + [8] + [4]
                fmapsizes = [256] + [128] + [64]
                channelDims= [128] + [128] + [128]
                query_index= [0, 1, 2]

            elif conf.cct_pos == 'source_encoder':
                self.key_candidates_idx = [10, 11, 13, 14, 16, 17] # for features in attention blocks
                patchsizes = [8]*2 + [4]*2 + [2]*2
                fmapsizes = [32]*2 + [16]*2 + [8]*2
                channelDims= [256]*2 + [512]*2 + [512]*2
                query_index= list(range(6))

            elif conf.cct_pos == 'source_encoder_inres_not': # not only for features in attention layer
                self.key_candidates_idx = [2, 5, 8, 11]
                patchsizes = [16] + [8] + [4] + [2]
                fmapsizes = [256] + [128] + [64] + [32]
                channelDims= [128]*2 + [256]*2
                query_index= list(range(4))
                

            self.cct_block = CCTConfig(patchSizes=patchsizes,
                                       fmapSizes= fmapsizes,
                                       channelDims= channelDims,
                                       query_index= query_index).make_model() 

        print("Pose_target_wo_target_encoder in AttentionBlock")

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        assert self.conf.is_stochastic
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample_z(self, n: int, device):
        assert self.conf.is_stochastic
        return torch.randn(n, self.conf.enc_out_channels, device=device)

    def noise_to_cond(self, noise: Tensor):
        raise NotImplementedError()
        assert self.conf.noise_net_conf is not None
        return self.noise_net.forward(noise)

    def encode(self, x):

        cond = self.encoder.forward(x)

        return {'cond': cond}

    @property
    def stylespace_sizes(self):
        modules = list(self.input_blocks.modules()) + list(
            self.middle_block.modules()) + list(self.output_blocks.modules())
        sizes = []
        for module in modules:
            if isinstance(module, ResBlock):
                linear = module.cond_emb_layers[-1]
                sizes.append(linear.weight.shape[0])
        return sizes

    def encode_stylespace(self, x, return_vector: bool = True):
        """
        encode to style space
        """
        modules = list(self.input_blocks.modules()) + list(
            self.middle_block.modules()) + list(self.output_blocks.modules())
        # (n, c)
        cond = self.encoder.forward(x)
        S = []
        for module in modules:
            if isinstance(module, ResBlock):
                # (n, c')
                s = module.cond_emb_layers.forward(cond)
                S.append(s)

        if return_vector:
            # (n, sum_c)
            return torch.cat(S, dim=1)
        else:
            return S


    def forward_with_cond_scale(
        self,
        x,
        t,
        cond,
        cond_scale,
        flow_fields,
        masks,
        **kwargs
    ):
        logits, flow_fields, feature_map_out = self.forward(x, t, cond=cond, prob = 1, feature_map_out= False, flow_fields= flow_fields, masks= masks, **kwargs) # cond: [model.encode(x_cond[0])['cond'], model.encode(torch.zeros_like(x_cond[0]))['cond']]

        if cond_scale == 1:
            return [logits, _, _]

        null_logits, _, _ = self.forward(x, t, cond=cond, prob = 0) # output the feature map
        
        if flow_fields is not None:
            flow_fields = [item.detach() for item in flow_fields]
        return [null_logits + (logits - null_logits) * cond_scale, logits, null_logits, flow_fields, feature_map_out]

    def forward(self,
                x,
                t,
                x_cond=None,
                prob=1,
                y=None,
                cond=None,
                style=None,
                noise=None,
                t_cond=None,
                feature_map_out= False,
                flow_fields= None,
                masks = None,
                **kwargs):
        """
        Apply the model to an input batch.

        Args:
            x_start: the original image to encode
            cond: output of the encoder
            noise: random noise (to predict the cond)
        """
        # x_cond = img = torch.cat([batch['source_image'], batch['target_image']], 0)
        # target_img = torch.cat([batch['target_image'], batch['source_image']], 0); 
        # x_start = target_img
        # x_t = self.q_sample(x_start, t, noise=noise)
        # target_pose = torch.cat([batch['target_skeleton'], batch['source_skeleton']], 0)
        # x = torch.cat([x_t, target_pose],1)
        _, target_pose = torch.split(x, [3, 20], dim= 1)
        # import pdb; pdb.set_trace()
        cond_mask = prob_mask_like((x.shape[0],), prob = prob, device = x.device)
        

        if t_cond is None:
            t_cond = t

        if noise is not None:
            # if the noise is given, we predict the cond from noise
            cond = self.noise_to_cond(noise)

        # below the code modifies the x_cond, so I generated the warp in advance.
        if self.conf.use_warp:
            if flow_fields is None and prob > 0:
                target_B, source_B = torch.chunk(target_pose, 2, dim= 0)
                source_pose = torch.cat([source_B, target_B], dim= 0)
            if x_cond is not None:
                flow_fields, masks = self.flow_net(x_cond, source_pose, target_pose) # source, source_B, target_B
            # len(flow_fields) == 2; flow_fields[0].shape: [4, 2, 32, 32]; flow_fields[1].shape: [4, 2, 64, 64]
            else:
                feature_maps = None
        else:
            flow_fields = None
            feature_maps = None
            
        flows_list, masks_list = [None for _ in np.arange(18)], [None for _ in np.arange(18)]

        if cond is None:
            x_cond  = (cond_mask.view(-1,1,1,1)*x_cond) # torch.Size([4, 1, 1, 1]) * torch.Size([4, 3, 256, 256])
            if x is not None:
                assert len(x) == len(x_cond), f'{len(x)} != {len(x_cond)}'

            tmp = self.encode(x_cond)
            cond = tmp['cond']
            # len(cond) == 18 == 6 * 3
            # each item of cond are: [4, 128, 256, 256] [4, 128, 128, 128] ... [4, 512, 8, 8]
            if self.conf.use_cct and self.conf.cct_pos == 'source_encoder_inres_not':
                key_candidates = self.source_enc_config(cond)
                query_candidates, attn_weights = self.cct_block(key_candidates)
                for q_idx, idx in enumerate(self.key_candidates_idx):
                    if idx in [10, 11, 13, 14, 16, 17]:
                        cond[idx] = query_candidates[q_idx]
        
            if self.conf.use_warp:
                
                # cond, flows_list, masks_list = self.GFLA_poseT_wo_target_enc_res3216(cond, flow_fields, masks)
                cond, flows_list, masks_list = self.GFLA_poseT_wo_target_enc_res32168(cond, flow_fields, masks)
                if feature_map_out:
                    feature_maps = self.out_fmap(cond)
                else:
                    feature_maps = None
            # input_sample = self.resample(source_vgg, flow).view(b, c, -1)
            else:

                if self.conf.ablation_pose_target_wo_target_enc:
                    cond, flows_list, masks_list = self.GFLA_poseT_wo_target_enc_ablation(cond)

        else:
            if prob==1:
                if self.conf.use_warp and type(cond).__name__ == 'list':
                    
                    step = kwargs.get('ddim_step')
                    cond[0], flows_list, masks_list = self.GFLA_poseT_wo_target_enc_res32168(cond[0], flow_fields, masks, ddim_step= step)                
                    cond = cond[0]

                    if feature_map_out:
                        feature_maps = self.out_fmap(cond)
                
                elif type(cond).__name__ == 'list':
                    cond = cond[0]
                    if self.conf.ablation_pose_target_wo_target_enc:
                        cond, flows_list, masks_list = self.GFLA_poseT_wo_target_enc_ablation(cond)

            elif prob==0:
                cond = cond[1]
            

        if t is not None:
            _t_emb = timestep_embedding(t, self.conf.model_channels) # torch.Size([4, 128])
            _t_cond_emb = timestep_embedding(t_cond, self.conf.model_channels) # torch.Size([4, 128])
        else:
            # this happens when training only autoenc
            _t_emb = None
            _t_cond_emb = None

        if self.conf.resnet_two_cond:
            res = self.time_embed.forward(
                time_emb=_t_emb,
                cond=cond,
                time_cond_emb=_t_cond_emb,
            ) # res.emb= cond(len(cond) == 18), res.time_emb= time_emb(128 -> 512; size= [4, 512]), res.style= cond(len(cond) == 10)
        else:
            raise NotImplementedError()

        if self.conf.resnet_two_cond:
            # two cond: first = time emb, second = cond_emb
            emb = res.time_emb # torch.size [4, 512]
            # cond_emb = res.emb
        else:
            # one cond = combined of both time and cond
            emb = res.emb
            cond_emb = None

        # override the style if given
        style = style or res.style # style == cond; len(style) == 18
        assert (y is not None) == (
            self.conf.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        if self.conf.num_classes is not None:
            raise NotImplementedError()
            # assert y.shape == (x.shape[0], )
            # emb = emb + self.label_emb(y)

        # where in the model to supply time conditions
        enc_time_emb = emb
        mid_time_emb = emb
        dec_time_emb = emb
        # where in the model to supply style conditions
        enc_cond_emb = cond
        mid_cond_emb = cond[-1] # torch.Size([4, 512, 8, 8])
        dec_cond_emb = cond #+ [cond[-1]]

        if self.conf.use_cct and self.conf.cct_pos == 'source_encoder':
            key_candidates = self.source_enc_config(cond)
            query_candidates, attn_weights = self.cct_block(key_candidates)
            for q_idx, idx in enumerate(self.key_candidates_idx):
                cond[idx] = query_candidates[q_idx]

        # hs = []
        hs = [[] for _ in range(len(self.conf.channel_mult))]

        if x is not None:
            h = x.type(self.dtype)

            # input blocks
            k = 0
            for i in range(len(self.input_num_blocks)):
                for j in range(self.input_num_blocks[i]):
                    if self.input_blocks[k] is not None:
                        h = self.input_blocks[k](h,
                                                 emb=enc_time_emb,
                                                 cond=enc_cond_emb[k],
                                                 flow_fields= flows_list[k],
                                                 masks = masks_list[k],
                                                 prob = prob
                                                )

                    hs[i].append(h if self.input_blocks[k] is not None else None)
                    #h = th.concat([h, enc_cond_emb[k]], 1)

                    k += 1



            assert k == len(self.input_blocks)

            # middle blocks
            h = self.middle_block(h, emb=mid_time_emb, cond=mid_cond_emb, flow_fields= flows_list[-1], masks = masks_list[-1], prob = prob)
        else:
            # no lateral connections
            # happens when training only the autonecoder
            h = None
            hs = [[] for _ in range(len(self.conf.channel_mult))]

        if self.conf.use_cct:
            if self.conf.cct_pos == 'input_blocks':
                key_candidates, key_coord = self.lateral_config(hs)
                query_candidates, attn_weights = self.cct_block(key_candidates)
                # id(key_candidates[idx]) == hs[idx][0] # idx == (0, 1, 2)
                for q_idx, indexes in enumerate(key_coord):
                    i, j = indexes
                    hs[i][j] = query_candidates[q_idx]
        
        # output blocks
        k = 0
        for i in range(len(self.output_num_blocks)):
            for j in range(self.output_num_blocks[i]):
                # take the lateral connection from the same layer (in reserve)
                # until there is no more, use None
                try:
                    lateral = hs[-i - 1].pop()
                    # print(i, j, lateral.shape)
                    if not self.conf.enable_architecture_remove and i in [len(self.output_num_blocks)-1, len(self.output_num_blocks)-2, len(self.output_num_blocks)-3] and j in self.conf.output_omit_block_id:
                        lateral = None
                except IndexError:
                    lateral = None
                    # print(i, j, lateral)

                if self.output_blocks[k] is not None:
                    h = self.output_blocks[k](h,
                                            emb=dec_time_emb,
                                            cond=dec_cond_emb[-k-1],
                                            lateral=lateral,
                                            flow_fields= flows_list[-k-1],
                                            masks = masks_list[-k-1],
                                            prob = prob)

                
                k += 1

        pred = self.out(h) # torch.Size([4, 6, 256, 256])

        return pred, flow_fields, feature_maps
    
    def cal_flow(self, flow_fields, cond):

        feature_warped_32 = [self.resample(cond[i], flow_fields[1]) for i in range(9, 12)] # for cond[9] - cond[11]
        feature_warped_16 = [self.resample(cond[i], flow_fields[0]) for i in range(12, 15)] # for cond[12] - cond[14]

        # flow upsampling or downsampling
        flow_8 = F.interpolate(flow_fields[0], [8, 8]) # 16 -> 8
        flow_64 = F.interpolate(flow_fields[1], [64, 64]) # 32 -> 64
        flow_128 = F.interpolate(flow_fields[1], [128, 128]) # 32 -> 128
        flow_256 = F.interpolate(flow_fields[1], [256, 256]) # 32 -> 256

        feature_warped_256 = [self.resample(cond[i], flow_256) for i in range(0, 3)]
        feature_warped_128 = [self.resample(cond[i], flow_128) for i in range(3, 6)]
        feature_warped_64 = [self.resample(cond[i], flow_64) for i in range(6, 9)]

        feature_warped_256.extend(feature_warped_128)
        feature_warped_256.extend(feature_warped_64)
        feature_warped_256.extend(feature_warped_32)
        feature_warped_256.extend(feature_warped_16)

        feature_warped_8 = [self.resample(cond[i], flow_8) for i in range(15, 18)]
        feature_warped_256.extend(feature_warped_8)
        del cond
        cond = feature_warped_256

        return cond
    
    def cal_flow64(self, flow_fields, cond):

        feature_warped_32 = [self.resample(cond[i], flow_fields[0]) for i in range(9, 12)] # for cond[9] - cond[11]
        feature_warped_64 = [self.resample(cond[i], flow_fields[1]) for i in range(6, 9)] # for cond[12] - cond[14]

        # flow upsampling or downsampling
        flow_8 = F.interpolate(flow_fields[0], [8, 8]) # 32 -> 8
        flow_16 = F.interpolate(flow_fields[0], [16, 16]) # 32 -> 16
        flow_128 = F.interpolate(flow_fields[1], [128, 128]) # 64 -> 128
        flow_256 = F.interpolate(flow_fields[1], [256, 256]) # 64 -> 256

        feature_warped_256 = [self.resample(cond[i], flow_256) for i in range(0, 3)]
        feature_warped_128 = [self.resample(cond[i], flow_128) for i in range(3, 6)]
        feature_warped_16 = [self.resample(cond[i], flow_16) for i in range(12, 15)]

        feature_warped_256.extend(feature_warped_128)
        feature_warped_256.extend(feature_warped_64)
        feature_warped_256.extend(feature_warped_32)
        feature_warped_256.extend(feature_warped_16)

        feature_warped_8 = [self.resample(cond[i], flow_8) for i in range(15, 18)]
        feature_warped_256.extend(feature_warped_8)
        cond = feature_warped_256

        return cond

    def cal_flow32_only(self, flow_fields, cond):

        # For effective cond index [10, 11, 13, 14, 16, 17]

        if len(flow_fields) == 1:
            feature_warped_32 = [self.resample(cond[i], flow_fields[0]) for i in range(10, 12)] # for cond[9] - cond[11]
        elif len(flow_fields) == 2:
            feature_warped_32 = [self.resample(cond[i], flow_fields[i-10]) for i in range(10, 12)]

        feature_warped_256 = [None for _ in range(10)]
        feature_16 = [None] + [cond[i] for i in range(13, 15)]
        feature_8 = [None] + [cond[i] for i in range(16, 18)]

        feature_warped_256.extend(feature_warped_32)
        feature_warped_256.extend(feature_16)
        feature_warped_256.extend(feature_8)

        return feature_warped_256

    def GFLA_poseT_block(self, target_B, cond, flow_fields, masks):

        """
        attn_layer= [2] for optical flow dimension is (64, 64)
        attn_layer= [3] for optical flow dimension is (32, 32)
        attn_layer= [4] for optical flow dimension is (16, 16)
        """
        # feature_list is a part of cond
        # feature_warped_32 = self.target(target_B, cond[10:12], flow_fields, masks) # for attn_layer= [3]
        feature_warped_32 = self.target(target_B, [cond[10]], flow_fields, masks) # for attn_layer= [3] and cond[10]
        feature_warped_32.append(feature_warped_32[0])

        feature_warped_256 = [None for _ in range(10)]
        # feature_16 = [None] + [cond[i] for i in range(13, 15)]
        feature_16 = [None] + [torch.cat([F.interpolate(feature_warped_32[0], [16, 16]) for _ in range(2)], dim= 1) for _ in range(2)]
        # feature_8 = [None] + [cond[i] for i in range(16, 18)]
        feature_8 = [None] + [torch.cat([F.interpolate(feature_warped_32[0], [8, 8]) for _ in range(2)], dim= 1) for _ in range(2)]

        feature_warped_256.extend(feature_warped_32)
        feature_warped_256.extend(feature_16)
        feature_warped_256.extend(feature_8)

        
        return feature_warped_256

    def GFLA_poseT_block_ablation(self, cond):

        feature_32 = [cond[10], cond[10]]

        feature_warped_256 = [None for _ in range(10)]
        # feature_16 = [None] + [cond[i] for i in range(13, 15)]
        feature_16 = [None] + [torch.cat([F.interpolate(feature_32[0], [16, 16]) for _ in range(2)], dim= 1) for _ in range(2)]
        # feature_8 = [None] + [cond[i] for i in range(16, 18)]
        feature_8 = [None] + [torch.cat([F.interpolate(feature_32[0], [8, 8]) for _ in range(2)], dim= 1) for _ in range(2)]

        feature_warped_256.extend(feature_32)
        feature_warped_256.extend(feature_16)
        feature_warped_256.extend(feature_8)
        
        return feature_warped_256

    def GFLA_poseT_wo_target_enc(self, cond, flow_fields: list, masks: list):

        # Only for the feature32 and flow 32

        feature_32 = [cond[10], cond[10]]
        flows_32 = [flow_fields[0], flow_fields[0]]
        masks_32 = [masks[0], masks[0]]

        feature_warped_256 = [None for _ in range(10)]
        flows_256, masks_256 = [None for _ in range(10)], [None for _ in range(10)]
        
        # feature_16 = [None] + [cond[i] for i in range(13, 15)]
        feature_16 = [None] + [torch.cat([F.interpolate(feature_32[0], [16, 16]) for _ in range(2)], dim= 1) for _ in range(2)]
        
        flows_16 = [None] + [F.interpolate(flow_fields[0], [16, 16]) for _ in range(2)]
        masks_16 = [None] + [F.interpolate(masks[0], [16, 16]) for _ in range(2)]
        
        # feature_8 = [None] + [cond[i] for i in range(16, 18)]
        feature_8 = [None] + [torch.cat([F.interpolate(feature_32[0], [8, 8]) for _ in range(2)], dim= 1) for _ in range(2)]
        
        flows_8 = [None] + [F.interpolate(flow_fields[0], [8, 8]) for _ in range(2)]
        masks_8 = [None] + [F.interpolate(masks[0], [8, 8]) for _ in range(2)]

        feature_warped_256.extend(feature_32)
        feature_warped_256.extend(feature_16)
        feature_warped_256.extend(feature_8)

        flows_256.extend(flows_32)
        flows_256.extend(flows_16)
        flows_256.extend(flows_8)

        masks_256.extend(masks_32)
        masks_256.extend(masks_16)
        masks_256.extend(masks_8)
        
        return feature_warped_256, flows_256, masks_256


    def GFLA_poseT_wo_target_enc_res3216(self, cond, flow_fields: list, masks: list, **kwargs):

        # For the resolution 32^2_sep, 16^2_sep fmap; len(flow_fields) == len(masks) == 2
        if self.conf.enable_source_enc_inres_remove and len(kwargs) == 0:
            cond[13] = cond[14]
            for idx in [10, 16]:
                assert cond[idx] is None
                cond[idx] = cond[idx-1]  

        elif self.conf.enable_source_enc_inres_remove and kwargs.get('ddim_step') == 0:
            cond[13] = cond[14]
            for idx in [10, 16]:
                assert cond[idx] is None
                cond[idx] = cond[idx-1]

        feature_32 = [cond[10], cond[11]]
        flows_32 = [flow_fields[1], flow_fields[1]]
        masks_32 = [masks[1], masks[1]]

        feature_warped_256 = [None for _ in range(10)]
        flows_256, masks_256 = [None for _ in range(10)], [None for _ in range(10)]
        
        # feature_16 = [None] + [cond[i] for i in range(13, 15)]
        feature_16 = [None] + [cond[13], cond[14]]
        
        flows_16 = [None] + [flow_fields[0], flow_fields[0]]
        masks_16 = [None] + [masks[0], masks[0]]
        
        # feature_8 = [None] + [cond[i] for i in range(16, 18)]
        feature_8 = [None] + [F.interpolate(feature_16[1], [8, 8]), F.interpolate(feature_16[2], [8, 8])]
        
        flows_8 = [None] + [F.interpolate(flow_fields[0], [8, 8]) for _ in range(2)]
        masks_8 = [None] + [F.interpolate(masks[0], [8, 8]) for _ in range(2)]

        feature_warped_256.extend(feature_32)
        feature_warped_256.extend(feature_16)
        feature_warped_256.extend(feature_8)

        flows_256.extend(flows_32)
        flows_256.extend(flows_16)
        flows_256.extend(flows_8)

        masks_256.extend(masks_32)
        masks_256.extend(masks_16)
        masks_256.extend(masks_8)
        
        return feature_warped_256, flows_256, masks_256

    
    def GFLA_poseT_wo_target_enc_res32168(self, cond, flow_fields: list, masks: list, **kwargs):

        # For the resolution 32^2_sep, 16^2_sep, 8^2_sep fmap; len(flow_fields) == len(masks) == 3
        if self.conf.enable_source_enc_inres_remove and len(kwargs) == 0:
            cond[13] = cond[14]
            for idx in [10, 16]:
                assert cond[idx] is None
                cond[idx] = cond[idx-1]  

        elif self.conf.enable_source_enc_inres_remove and kwargs.get('ddim_step') == 0:
            cond[13] = cond[14]
            for idx in [10, 16]:
                assert cond[idx] is None
                cond[idx] = cond[idx-1]

        feature_32 = [cond[10], cond[11]]
        flows_32 = [flow_fields[2], flow_fields[2]]
        masks_32 = [masks[2], masks[2]]

        feature_warped_256 = [None for _ in range(10)]
        flows_256, masks_256 = [None for _ in range(10)], [None for _ in range(10)]
        
        # feature_16 = [None] + [cond[i] for i in range(13, 15)]
        # feature_16 = [None] + [cond[13], cond[14]]
        # feature_16 = [None] + [cond[13], cond[13]]
        feature_16 = [None] + [cond[13], cond[14]]
        
        flows_16 = [None] + [flow_fields[1], flow_fields[1]]
        masks_16 = [None] + [masks[1], masks[1]]
        
        # feature_8 = [None] + [cond[i] for i in range(16, 18)]
        # feature_8 = [None] + [F.interpolate(feature_16[1], [8, 8]), F.interpolate(feature_16[2], [8, 8])]
        feature_8 = [None] + [cond[16], cond[17]]
        
        # flows_8 = [None] + [F.interpolate(flow_fields[0], [8, 8]) for _ in range(2)]
        flows_8 = [None] + [flow_fields[0], flow_fields[0]]
        # masks_8 = [None] + [F.interpolate(masks[0], [8, 8]) for _ in range(2)]
        masks_8 = [None] + [masks[0], masks[0]]

        feature_warped_256.extend(feature_32)
        feature_warped_256.extend(feature_16)
        feature_warped_256.extend(feature_8)

        flows_256.extend(flows_32)
        flows_256.extend(flows_16)
        flows_256.extend(flows_8)

        masks_256.extend(masks_32)
        masks_256.extend(masks_16)
        masks_256.extend(masks_8)
        
        return feature_warped_256, flows_256, masks_256


    def GFLA_poseT_wo_target_enc_ablation(self, cond):
        
        b, _, _, _ = cond[10].shape
        feature_32 = [cond[10], cond[10]]
        flows_32 = [torch.zeros(b, 2, 32, 32, dtype= torch.float32).cuda(), torch.zeros(b, 2, 32, 32, dtype= torch.float32).cuda()]
        masks_32 = [torch.zeros(b, 1, 32, 32, dtype= torch.float32).cuda(), torch.zeros(b, 1, 32, 32, dtype= torch.float32).cuda()]

        feature_warped_256 = [None for _ in range(10)]
        flows_256, masks_256 = [None for _ in range(10)], [None for _ in range(10)]
        
        # feature_16 = [None] + [cond[i] for i in range(13, 15)]
        feature_16 = [None] + [torch.cat([F.interpolate(feature_32[0], [16, 16]) for _ in range(2)], dim= 1) for _ in range(2)]
        
        flows_16 = [None] + [torch.zeros(b, 2, 16, 16, dtype= torch.float32).cuda() for _ in range(2)]
        masks_16 = [None] + [torch.zeros(b, 1, 16, 16, dtype= torch.float32).cuda() for _ in range(2)]
        
        # feature_8 = [None] + [cond[i] for i in range(16, 18)]
        feature_8 = [None] + [torch.cat([F.interpolate(feature_32[0], [8, 8]) for _ in range(2)], dim= 1) for _ in range(2)]
        
        flows_8 = [None] + [torch.zeros(b, 2, 8, 8, dtype= torch.float32).cuda() for _ in range(2)]
        masks_8 = [None] + [torch.zeros(b, 1, 8, 8, dtype= torch.float32).cuda() for _ in range(2)]

        feature_warped_256.extend(feature_32)
        feature_warped_256.extend(feature_16)
        feature_warped_256.extend(feature_8)

        flows_256.extend(flows_32)
        flows_256.extend(flows_16)
        flows_256.extend(flows_8)

        masks_256.extend(masks_32)
        masks_256.extend(masks_16)
        masks_256.extend(masks_8)
        
        return feature_warped_256, flows_256, masks_256

    
    def out_fmap(self, cond: List[int]):
        feature_maps = [] # Add feature maps
        for index in [11, 14, 17]:
            if index in [14, 17]:
                upsample_map = F.interpolate(cond[index].detach(), [32, 32]) # [B, 512, 32, 32]
                batch_features = []
                for i in range(upsample_map.shape[0]):
                    gray_scale = torch.sum(upsample_map[i],0) # [32, 32]
                    gray_scale = gray_scale / upsample_map[i].shape[0] # [32, 32]
                    gray_scale = gray_scale.unsqueeze(0) # # [1, 32, 32]
                    batch_features.append(gray_scale)
                                
            else:
                upsample_map = cond[index].detach() # [B, 256, 32, 32]
                batch_features = []
                for i in range(upsample_map.shape[0]):
                    gray_scale = torch.sum(upsample_map[i],0) # [32, 32]
                    gray_scale = gray_scale / upsample_map[i].shape[0] # [32, 32]
                    gray_scale = gray_scale.unsqueeze(0) # # [1, 32, 32]
                    batch_features.append(gray_scale)

            feature_maps.append(torch.cat(batch_features, dim= 0))
        
        return feature_maps

    def lateral_config(self, hs):

        key_candidates = []
        key_candidate_coord = []
        enough_num = False

        for i in range(len(hs)):
            for j in range(len(hs[i])):
                if hs[i][j] is not None:
                    key_candidates.append(hs[i][j])
                    key_candidate_coord.append((i, j))
                
                if len(key_candidates) == self.key_candidates_num:
                    enough_num = True
                    break
            
            if enough_num:
                break
        
        return key_candidates, key_candidate_coord

    def source_enc_config(self, cond):
        key_candidates = [cond[idx] for idx in self.key_candidates_idx if cond[idx] is not None]

        return key_candidates


class AutoencReturn(NamedTuple):
    pred: Tensor
    cond: Tensor = None


class EmbedReturn(NamedTuple):
    # style and time
    emb: Tensor = None
    # time only
    time_emb: Tensor = None
    # style only (but could depend on time)
    style: Tensor = None


class TimeStyleSeperateEmbed(nn.Module):
    # embed only style
    def __init__(self, time_channels, time_out_channels):
        super().__init__()
        self.time_embed = nn.Sequential(
            linear(time_channels, time_out_channels),
            nn.SiLU(),
            linear(time_out_channels, time_out_channels),
        )
        self.style = nn.Identity()

    def forward(self, time_emb=None, cond=None, **kwargs):
        if time_emb is None:
            # happens with autoenc training mode
            time_emb = None
        else:
            time_emb = self.time_embed(time_emb)
        style = self.style(cond)
        return EmbedReturn(emb=style, time_emb=time_emb, style=style)
