import torch
import torch.nn as nn
import torch.nn.functional as functional
from model import model_utils


# LF depth estimator
class LF_depth_estimator(nn.Module):
    def __init__(self, cfg):
        super(LF_depth_estimator, self).__init__()

        an2 = cfg.angular_resolution * cfg.angular_resolution

        self.depth_block1 = nn.Sequential(
            nn.Conv2d(in_channels=an2, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True))

        self.depth_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True))

        self.depth_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Tanh())

    def forward(self, lfi):
        depth_features = self.depth_block1(lfi)
        depth_features = self.depth_block2(depth_features)
        depth = self.depth_block3(depth_features)
        return depth