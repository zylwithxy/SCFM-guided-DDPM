from models.networks.base_function import *
from models.networks.base_network import BaseNetwork
from dataclasses import dataclass
from typing import List
from models.networks.resample2d_package.resample2d import Resample2d
import torch.nn.functional as F
from models.networks.correlation_package.correlation import Correlation

@dataclass
class PoseTargetConfig():
    # number of style channels
    image_nc: int = 3
    structure_nc: int = 20 #18
    ngf: int = 64
    img_f: int = 512
    layers: int = 3
    num_blocks: int = 2
    norm: str = 'instance'
    activation: str = 'LeakyReLU'
    attn_layer: List[int] = None # list [2, 3]
    extractor_kz: dict = None # {'2': 5, '3': 3}, for not conf.use_encoder {'3': 3, '1': 2, '2': 1}
    
    use_spect: bool = False
    use_coord: bool = False

    use_encoder: bool = True # if use the encoder to encode features

    use_flowdisp: bool = False # if use the displacement map and confidence map for optical flow

    flowdisp_inputdim: int = 0 # input dimension for the main feat for optical flow
    resolution: int = 0 # for the resolution of the feature map

    def make_model(self):
        return PoseTargetNet(self)


class PoseTargetNet(BaseNetwork):
    def __init__(self, conf: PoseTargetConfig):  
        super(PoseTargetNet, self).__init__()

        self.conf = conf
        self.layers = conf.layers
        self.attn_layer = conf.attn_layer

        norm_layer = get_norm_layer(norm_type= conf.norm)
        nonlinearity = get_nonlinearity_layer(activation_type=conf.activation)


        if conf.use_encoder:
            self.block0 = EncoderBlock(conf.structure_nc, conf.ngf, norm_layer,
                                 nonlinearity, conf.use_spect, conf.use_coord)
            mult = 1
            for i in range(conf.layers-1):
                mult_prev = mult
                mult = min(2 ** (i + 1), conf.img_f//conf.ngf)
                block = EncoderBlock(conf.ngf*mult_prev, conf.ngf*mult, norm_layer,
                                 nonlinearity, conf.use_spect, conf.use_coord)
                setattr(self, 'encoder' + str(i), block)         


        # decoder part
        mult = min(2 ** (conf.layers-1), conf.img_f//conf.ngf)
        for i in range(conf.layers):
            mult_prev = mult
            mult = min(2 ** (conf.layers-i-2), conf.img_f//conf.ngf) if i != conf.layers-1 else 1
            # if conf.num_blocks == 1:
            #     up = nn.Sequential(ResBlockDecoder(conf.ngf*mult_prev, conf.ngf*mult, None, norm_layer, 
            #                              nonlinearity, conf.use_spect, conf.use_coord))
            # else:
            #     up = nn.Sequential(ResBlocks(conf.num_blocks-1, conf.ngf*mult_prev, None, None, norm_layer, 
            #                                  nonlinearity, False, conf.use_spect, conf.use_coord),
            #                        ResBlockDecoder(conf.ngf*mult_prev, conf.ngf*mult, None, norm_layer, 
            #                                  nonlinearity, conf.use_spect, conf.use_coord))
            # setattr(self, 'decoder' + str(i), up)

            if conf.layers-i in conf.attn_layer:
                attn = ExtractorAttn(conf.ngf*mult_prev, conf.extractor_kz[str(conf.layers-i)], nonlinearity, softmax=True)
                setattr(self, 'attn' + str(i), attn)

        if not conf.use_encoder:

            for i in range(1, 3):
                attn = ExtractorAttn(512, 3 - i, nonlinearity, softmax=True)
                setattr(self, 'attn' + str(i), attn) # self.attn1 [16^2]; self.attn2 [8^2]

        if conf.use_flowdisp:
            self.backwarp = Resample2d(4, 1, sigma=2)
            self.dispMain = nn.Sequential(
                    nn.Conv2d(in_channels= conf.flowdisp_inputdim + 3, out_channels=128, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
            self.confNet = nn.Sequential(
                        nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
                        nn.Sigmoid()
                    )

            index_pad = (32, 16, 8,).index(conf.resolution)
            pad_list = [4, 3, 2]
            confFeat_list = [25, 9, 9]

            self.autoCorr = Correlation(pad_size=pad_list[index_pad], kernel_size=1, max_displacement=pad_list[index_pad], stride1=1, stride2=2)

            self.confFeat = nn.Sequential(
                        nn.Conv2d(in_channels= confFeat_list[index_pad] + 1, out_channels=128, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    )
            
            self.dispNet = nn.Sequential(
                        nn.Conv2d(in_channels=32, out_channels=2, kernel_size=5, stride=1, padding=2)
                        )
        

        # self.outconv = Output(ngf, output_nc, 3, None, nonlinearity, use_spect, use_coord)


    def forward(self, target_B: torch.Tensor, source_feature: List, flow_fields: List, masks: List):

        assert len(source_feature) == len(flow_fields) == len(masks), 'The number of source_feature, flow_fields and masks should be same'

        if self.conf.use_encoder:
            out = self.block0(target_B)
            for i in range(self.layers-1):
                model = getattr(self, 'encoder' + str(i))
                out = model(out)
        else:
            out = target_B # target_B features

        if self.conf.use_flowdisp:
            bs = flow_fields[0].shape[0]
            tenDiff = (out - self.backwarp(source_feature[0], flow_fields[0])).pow(2.0).sum(1, True).sqrt().detach()
            mainfeat = self.dispMain(torch.cat([tenDiff, 
                                                flow_fields[0] - flow_fields[0].view(bs, 2, -1).mean(2, True).view(bs, 2, 1, 1), 
                                                out], 1))

            tenConf = self.confNet(mainfeat) # confidence map
            tenCorrelation = F.leaky_relu(input=self.autoCorr(out, out), 
                                          negative_slope=0.1, inplace=False)
            confFeat = self.confFeat(torch.cat([tenCorrelation, tenConf], 1))
            tenDisp = self.dispNet(confFeat)

            flow_fields[0] = self.backwarp(flow_fields[0], tenDisp)



        counter = 0
        out_features = [] 

        if source_feature[0].shape[2] == 32:
            idx = 0
        elif source_feature[0].shape[2] == 16:
            idx = 1
        elif source_feature[0].shape[2] == 8:
            idx = 2    

        model = getattr(self, 'attn' + str(idx))
        out_attn = model(source_feature[counter], out, flow_fields[counter])      
        out = out*(1-masks[counter]) + out_attn*masks[counter]
        out_features.append(out)
            
            #----------------------------Modify above-------------------------------------#

            # model = getattr(self, 'decoder' + str(i))
            # out = model(out)

        # out_image = self.outconv(out)

        return out_features

    def forward_hook_function(self, target_B, source_feature, flow_fields, masks):
        hook_target=[]
        hook_source=[]      
        hook_attn=[]      
        hook_mask=[]      
        out = self.block0(target_B)
        for i in range(self.layers-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out) 

        counter=0
        for i in range(self.layers):
            if self.layers-i in self.attn_layer:
                model = getattr(self, 'attn' + str(i))

                attn_param, out_attn = model.hook_attn_param(source_feature[i], out, flow_fields[counter])        
                out = out*(1-masks[counter]) + out_attn*masks[counter]

                hook_target.append(out)
                hook_source.append(source_feature[i])
                hook_attn.append(attn_param)
                hook_mask.append(masks[counter])
                counter += 1

            model = getattr(self, 'decoder' + str(i))
            out = model(out)

        out_image = self.outconv(out)
        return hook_target, hook_source, hook_attn, hook_mask