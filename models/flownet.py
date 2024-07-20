import torch.nn as nn
from models.networks.base_function import *
from dataclasses import dataclass
from typing import List

@dataclass
class FlowNetConfig():
    # number of style channels
    image_nc: int = 3
    structure_nc: int = 20 #18
    ngf: int = 32
    img_f: int = 256
    encoder_layer: int = 5
    attn_layer: List[int] = None # list
    norm: str = 'instance'
    activation: str = 'LeakyReLU'
    use_spect: bool = False
    use_coord: bool = False
    use_warp_separate: bool = False # generate the same resolution warp separately, for example (32, 32)

    def make_model(self):
        return PoseFlowNet(self)


class PoseFlowNet(nn.Module):
    """docstring for FlowNet"""
    def __init__(self, conf: FlowNetConfig):
        super(PoseFlowNet, self).__init__()

        self.conf = conf
        self.encoder_layer = conf.encoder_layer
        self.decoder_layer = conf.encoder_layer - min(conf.attn_layer)
        self.attn_layer = conf.attn_layer
        norm_layer = get_norm_layer(norm_type=conf.norm)
        nonlinearity = get_nonlinearity_layer(activation_type=conf.activation)
        input_nc = 2*conf.structure_nc + conf.image_nc

        self.block0 = EncoderBlock(input_nc, conf.ngf, norm_layer,
                                 nonlinearity, conf.use_spect, conf.use_coord)
        mult = 1
        for i in range(conf.encoder_layer-1):
            mult_prev = mult
            mult = min(2 ** (i + 1), conf.img_f//conf.ngf)
            block = EncoderBlock(conf.ngf*mult_prev, conf.ngf*mult,  norm_layer,
                                 nonlinearity, conf.use_spect, conf.use_coord)
            setattr(self, 'encoder' + str(i), block)         
        
        for i in range(self.decoder_layer):
            mult_prev = mult
            mult = min(2 ** (conf.encoder_layer-i-2), conf.img_f//conf.ngf) if i != conf.encoder_layer-1 else 1
            up = ResBlockDecoder(conf.ngf*mult_prev, conf.ngf*mult, conf.ngf*mult, norm_layer, 
                                    nonlinearity, conf.use_spect, conf.use_coord)
            setattr(self, 'decoder' + str(i), up)
            
            jumpconv = Jump(conf.ngf*mult, conf.ngf*mult, 3, None, nonlinearity, conf.use_spect, conf.use_coord)
            setattr(self, 'jump' + str(i), jumpconv)

            if conf.encoder_layer-i-1 in conf.attn_layer:
                flow_out = nn.Conv2d(conf.ngf*mult, 2, kernel_size=3,stride=1,padding=1,bias=True)
                setattr(self, 'output' + str(i), flow_out)

                flow_mask = nn.Sequential(nn.Conv2d(conf.ngf*mult, 1, kernel_size=3,stride=1,padding=1,bias=True),
                                          nn.Sigmoid())
                setattr(self, 'mask' + str(i), flow_mask)
            
            if conf.encoder_layer-i-1 in conf.attn_layer and conf.use_warp_separate:
                flow_out = nn.Conv2d(conf.ngf*mult, 2, kernel_size=3,stride=1,padding=1,bias=True)
                setattr(self, 'output_copy' + str(i), flow_out)

                flow_mask = nn.Sequential(nn.Conv2d(conf.ngf*mult, 1, kernel_size=3,stride=1,padding=1,bias=True),
                                          nn.Sigmoid())
                setattr(self, 'mask_copy' + str(i), flow_mask)
        
        if 5 in conf.attn_layer: # dim of flow map (b, 2, 8, 8)
            self.output_eight = nn.Conv2d(256, 2, kernel_size=3,stride=1,padding=1,bias=True)
            self.mask_eight = nn.Sequential(nn.Conv2d(256, 1, kernel_size=3,stride=1,padding=1,bias=True),
                                          nn.Sigmoid())
            if conf.use_warp_separate:
                self.output_eight_copy = nn.Conv2d(256, 2, kernel_size=3,stride=1,padding=1,bias=True)
                self.mask_eight_copy = nn.Sequential(nn.Conv2d(256, 1, kernel_size=3,stride=1,padding=1,bias=True),
                                          nn.Sigmoid())
            

    def forward(self, source, source_B, target_B):
        flow_fields=[]
        masks=[]
        inputs = torch.cat((source, source_B, target_B), 1) 
        out = self.block0(inputs)
        result=[out]
        for i in range(self.encoder_layer-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            result.append(out)
        
        if 5 in self.attn_layer:
            flow_field, mask = self.flow_mask_eight_output(out, '')
            flow_fields.append(flow_field)
            masks.append(mask)
            if self.conf.use_warp_separate:
                flow_field, mask = self.flow_mask_eight_output(out, '_copy')
                flow_fields.append(flow_field)
                masks.append(mask)

        for i in range(self.decoder_layer):
            model = getattr(self, 'decoder' + str(i))
            out = model(out)

            model = getattr(self, 'jump' + str(i))
            jump = model(result[self.encoder_layer-i-2])
            out = out+jump

            if self.encoder_layer-i-1 in self.attn_layer:
                flow_field, mask = self.attn_output(out, i)
                flow_fields.append(flow_field)
                masks.append(mask)

                if self.conf.use_warp_separate:
                    flow_field, mask = self.flow_mask_sep_output(out, i)
                    flow_fields.append(flow_field)
                    masks.append(mask)
                

        return flow_fields, masks

    def attn_output(self, out, i):
        model = getattr(self, 'output' + str(i))
        flow = model(out)
        model = getattr(self, 'mask' + str(i))
        mask = model(out)
        return flow, mask  
    
    def flow_mask_sep_output(self, out, i):
        model = getattr(self, 'output_copy' + str(i))
        flow = model(out)
        model = getattr(self, 'mask_copy' + str(i))
        mask = model(out)

        return flow, mask
    
    def flow_mask_eight_output(self, out, suffix: str):
        model = getattr(self, 'output_eight' + suffix)
        flow = model(out)
        model = getattr(self, 'mask_eight' + suffix)
        mask = model(out)

        return flow, mask