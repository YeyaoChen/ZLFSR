import torch
import torch.nn as nn
import torch.nn.functional as functional
from model import model_utils


# Optical flow estimator
class Optical_flow_estimator(nn.Module):
    def __init__(self, DFs):
        super(Optical_flow_estimator, self).__init__()

        self.optical_flow_estimate = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=DFs[0], dilation=DFs[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=DFs[1], dilation=DFs[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=DFs[2], dilation=DFs[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=DFs[3], dilation=DFs[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=DFs[4], dilation=DFs[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=DFs[5], dilation=DFs[5]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=DFs[6], dilation=DFs[6]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=DFs[7], dilation=DFs[7]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=4, kernel_size=3, stride=1, padding=DFs[8], dilation=DFs[8]),
            nn.Tanh())

    def forward(self, img_pair):
        optical_flow = self.optical_flow_estimate(img_pair)
        return optical_flow