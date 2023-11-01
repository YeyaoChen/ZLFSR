import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
import numpy as np
from math import exp
from model import Vgg19


# Functions
def gradient_cal(in_img):
    grad1 = in_img[:, :, 1:, :] - in_img[:, :, :-1, :]     # [b,an2,h,w]
    grad2 = in_img[:, :, :, 1:] - in_img[:, :, :, :-1]
    return grad1, grad2


def lfi2epi(lfi):
    B, an2, h, w = lfi.shape
    an = int(np.sqrt(an2))
    # [B,an2,h,w] -> [B*ah*h,aw,w]  &  [B*aw*w,ah,h]
    epi_h = lfi.view(B, an, an, h, w).permute(0, 1, 3, 2, 4).contiguous().view(-1, 1, an, w)
    epi_v = lfi.view(B, an, an, h, w).permute(0, 2, 4, 1, 3).contiguous().view(-1, 1, an, h)
    return epi_h, epi_v


def L1_loss(X, Y):
    eps = 1e-6
    diff = torch.add(X, -Y)
    error = torch.sqrt(diff * diff + eps)
    Charbonnier_loss = torch.sum(error) / torch.numel(error)
    return Charbonnier_loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = functional.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = functional.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = functional.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = functional.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = functional.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


#################### Loss functions ####################
class L1_Reconstruction_loss(nn.Module):
    def __init__(self):
        super(L1_Reconstruction_loss, self).__init__()

    def forward(self, infer, ref):
        rec_loss = L1_loss(infer, ref)   # [b,an2,h,w]
        return rec_loss


class Perceptual_loss(nn.Module):
    def __init__(self, device):
        super(Perceptual_loss, self).__init__()
        self.device = device

    def forward(self, infer, ref):
        vgg19 = Vgg19.Vgg19(requires_grad=False).to(self.device)
        infer_relu = vgg19(infer)
        ref_relu = vgg19(ref)

        loss1_2 = L1_loss(infer_relu[0], ref_relu[0])
        loss2_2 = L1_loss(infer_relu[1], ref_relu[1])
        loss3_2 = L1_loss(infer_relu[2], ref_relu[2])
        loss4_2 = L1_loss(infer_relu[3], ref_relu[3])
        loss5_2 = L1_loss(infer_relu[4], ref_relu[4])
        per_loss = (loss1_2 + loss2_2 + loss3_2 + loss4_2 + loss5_2) / 5.0
        return per_loss


class Smooth_loss(nn.Module):
    def __init__(self):
        super(Smooth_loss, self).__init__()

    def forward(self, in_depth, in_img):
        depth_grad_x, depth_grad_y = gradient_cal(in_depth)     # [B,c,h,w]
        img_grad_x, img_grad_y = gradient_cal(in_img)           # [B,c,h,w]

        weights_x = torch.exp(-1.0 * torch.abs(img_grad_x))     # [B,c,h,w]
        weights_y = torch.exp(-1.0 * torch.abs(img_grad_y))

        smoothness_x = depth_grad_x * weights_x
        smoothness_y = depth_grad_y * weights_y
        smooth_loss = smoothness_x.abs().mean() + smoothness_y.abs().mean()
        return smooth_loss


class EPIGrad_loss(nn.Module):
    def __init__(self):
        super(EPIGrad_loss, self).__init__()

    def forward(self, infer, ref):
        infer_epi_h, infer_epi_v = lfi2epi(infer)
        infer_dx_h, infer_dy_h = gradient_cal(infer_epi_h)
        infer_dx_v, infer_dy_v = gradient_cal(infer_epi_v)

        ref_epi_h, ref_epi_v = lfi2epi(ref)
        ref_dx_h, ref_dy_h = gradient_cal(ref_epi_h)
        ref_dx_v, ref_dy_v = gradient_cal(ref_epi_v)
        epi_loss_h = L1_loss(infer_dx_h, ref_dx_h) + L1_loss(infer_dy_h, ref_dy_h)
        epi_loss_v = L1_loss(infer_dx_v, ref_dx_v) + L1_loss(infer_dy_v, ref_dy_v)
        epi_loss = epi_loss_h + epi_loss_v
        return epi_loss


class Detail_loss(nn.Module):
    def __init__(self):
        super(Detail_loss, self).__init__()

    def forward(self, infer, ref):
        infer_dx, infer_dy = gradient_cal(infer)
        ref_dx, ref_dy = gradient_cal(ref)
        detail_loss = L1_loss(infer_dx, ref_dx) + L1_loss(infer_dy, ref_dy)
        return detail_loss


class SSIM_loss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM_loss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channels = 1
        self.window = create_window(window_size, self.channels)

    def forward(self, infer, ref):
        batchsize, angres2, height, width = infer.shape           # [b,an2,h,w]
        infer = infer.view(batchsize*angres2, 1, height, width)   # [b*an2,1,h,w]
        ref = ref.view(batchsize*angres2, 1, height, width)       # [b*an2,1,h,w]

        _, channels, _, _ = infer.shape    # [b,an2,h,w]
        if channels == self.channels and self.window.data.type() == infer.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channels)

            if infer.is_cuda:
                window = window.cuda(infer.get_device())
            window = window.type_as(infer)

            self.window = window
            self.channels = channels

        ssim_value = _ssim(infer, ref, window, self.window_size, channels, self.size_average)
        return 1.0 - ssim_value


def get_loss(opt, device):
    losses = {}
    if (opt.l1_weight > 0):
        losses['l1_loss'] = L1_Reconstruction_loss()

    if (opt.ssim_weight > 0):
        losses['ssim_loss'] = SSIM_loss()

    if (opt.perceptual_weight > 0):
        losses['perceptual_loss'] = Perceptual_loss(device)

    if (opt.smooth_weight > 0):
        losses['smooth_loss'] = Smooth_loss()

    if (opt.epigrad_weight > 0):
        losses['epigrad_loss'] = EPIGrad_loss()

    if (opt.detail_weight > 0):
        losses['detail_loss'] = Detail_loss()
    return losses