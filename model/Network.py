import torch.nn as nn
import torch.nn.functional as F
import torch

from .SegNeXt_Encoder import Mscan_Encoder
from .Segmentation_Decoder import Seg_Decoder
from .SegNeXt_Encoder import get_mscan_model_shape

class Network(nn.Module):
    def __init__(self, cfg):
        super(Network, self).__init__()
        self.encoder = Mscan_Encoder(cfg)
        self.polyp_seg_decoder = Seg_Decoder(cfg)

    def forward(self, img):
        features = self.encoder(img)
        outs = self.polyp_seg_decoder(features)

        return outs


if __name__ == '__main__':
    import torch
    from option import args

    img = torch.randn(16, 3, 320, 320).cuda()
    model = Network(args).cuda()

    while True:
        out = model(img, flag='polyp_seg')
        print(out[0].size())
        out = model(img, flag='highlight_seg')
        print(out[0].size())
        out = model(img, flag='highlight_removal')
        print(out.size())
