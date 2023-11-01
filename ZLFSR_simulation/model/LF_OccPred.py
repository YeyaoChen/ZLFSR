import torch
import torch.nn as nn
import torch.nn.functional as functional
from model import model_utils


# LF Occlusion prediction
class LF_OccPred(nn.Module):
    def __init__(self, cfg):
        super(LF_OccPred, self).__init__()

        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True))

        self.sas_layers = model_utils.make_SASlayers(SAS_num=3, angres=cfg.angular_resolution, channel=64)

        self.tail_conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1))

    def forward(self, in_lfi):
        lf_feats = self.head_conv(in_lfi)        # [b*an2,1,h,w]
        lf_feats = self.sas_layers(lf_feats)     # [b*an2,c,h,w]
        res_lfi = self.tail_conv(lf_feats)       # [b*an2,1,h,w]
        return in_lfi + res_lfi
