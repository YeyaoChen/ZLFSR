import torch
import torch.nn as nn
import torch.nn.functional as functional


class SAS_layer(nn.Module):
    def __init__(self, an, chan):
        super(SAS_layer, self).__init__()

        self.an = an
        self.an2 = an * an
        self.spaconv1 = nn.Conv2d(in_channels=chan, out_channels=chan, kernel_size=3, stride=1, padding=1, dilation=1)
        self.spaconv2 = nn.Conv2d(in_channels=chan, out_channels=chan, kernel_size=3, stride=1, padding=1, dilation=1)
        self.angconv = nn.Conv2d(in_channels=chan, out_channels=chan, kernel_size=3, stride=1, padding=1, dilation=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, in_feats):
        Ban, c, h, w = in_feats.shape    # [B*an2,c,h,w]
        Bs = Ban//self.an2

        # Spatial convolution
        conv_feats = self.relu(self.spaconv1(in_feats))      # [B*an2,c,h,w]
        conv_feats = self.relu(self.spaconv2(conv_feats))    # [B*an2,c,h,w]

        # Spatial dimension ---> Angular dimension
        conv_feats = conv_feats.reshape(Bs, self.an2, c, h*w)            # [B,an2,c,h*w]
        conv_feats = torch.transpose(conv_feats, 1, 3)                   # [B,h*w,c,an2]
        conv_feats = conv_feats.reshape(Bs*h*w, c, self.an, self.an)     # [B*h*w,c,an,an]

        # Angular convolution
        conv_feats = self.relu(self.angconv(conv_feats))     # [B*h*w,c,an,an]

        # Angular dimension ---> Spatial dimension
        conv_feats = conv_feats.reshape(Bs, h*w, c, self.an2)       # [B,h*w,c,an2]
        conv_feats = torch.transpose(conv_feats, 1, 3)              # [B,an2,c,h*w]
        conv_feats = conv_feats.reshape(Bs*self.an2, c, h, w)       # [B*an2,c,h,w]
        return conv_feats


def make_SASlayers(SAS_num, angres, channel):
    SAS_layers = []
    for si in range(SAS_num):
        SAS_layers.append(SAS_layer(angres, channel))
    return nn.Sequential(*SAS_layers)