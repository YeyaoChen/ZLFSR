import torch
import torch.nn as nn
import numpy as np
import math
import copy
from scipy.signal import convolve2d
import pandas as pd
import os
import imageio


###################### Functions ######################
def ycbcr2rgb(in_ycbcr):
    color_matrix = np.array([[65.481, 128.553, 24.966],
                             [-37.797, -74.203, 112],
                             [112, -93.786, -18.214]])
    in_shape = in_ycbcr.shape
    if len(in_shape) == 3:
        in_ycbcr = in_ycbcr.reshape((in_shape[0] * in_shape[1], 3))
    out_rgb = copy.deepcopy(in_ycbcr)
    out_rgb[:, 0] -= 16. / 255.
    out_rgb[:, 1:] -= 128. / 255.
    out_rgb = np.dot(out_rgb, np.linalg.inv(color_matrix.transpose()) * 255.)
    out_rgb = out_rgb.clip(0, 1).reshape(in_shape).astype(np.float32)
    return out_rgb


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    pixel_max = 1.0
    return 10 * math.log10(pixel_max / mse)


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    g = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    g[g < np.finfo(g.dtype).eps * g.max()] = 0
    sumg = g.sum()
    if sumg != 0:
        g /= sumg
    return g


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def calculate_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=1.0):
    if not im1.shape == im2.shape:
        raise ValueError("Input Images must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'same')
    mu2 = filter2(im2, window, 'same')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'same') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'same') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'same') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(np.mean(ssim_map))


def calculate_total_scores(pred_y, gt_y, out_ang2):
    view_list = []
    view_psnr_y = []
    view_ssim_y = []

    for ii in range(out_ang2):
        if ii != (out_ang2//2):    # central SAI
            cur_pred_y = pred_y[0, ii]   # [h,w]
            cur_gt_y = gt_y[0, ii]       # [h,w]

            cur_psnr_y = calculate_psnr(cur_pred_y, cur_gt_y)
            cur_ssim_y = calculate_ssim(cur_pred_y, cur_gt_y)

            view_list.append(ii)
            view_psnr_y.append(cur_psnr_y)
            view_ssim_y.append(cur_ssim_y)
    return np.mean(view_psnr_y), np.mean(view_ssim_y)


def save_LFimg(pred_ycbcr, save_path, lfi_no, out_ang, isWarp=True):
    # pred_ycbcr: [b,an2,h,w,3]
    save_warp_dir = save_path + str(lfi_no)
    save_OccPred_dir = save_path + str(lfi_no)
    if not os.path.exists(save_warp_dir):
        os.makedirs(save_warp_dir)
    if not os.path.exists(save_OccPred_dir):
        os.makedirs(save_OccPred_dir)

    for t in range(out_ang * out_ang):
        ycbcr_im = pred_ycbcr[0, t]           # [1,1,h,w,3]
        rgb_im = ycbcr2rgb(ycbcr_im)
        rgb_im = (rgb_im.clip(0, 1) * 255.0).astype(np.uint8)
        u = t // out_ang
        v = t % out_ang
        if isWarp:
            img_name = '{}/view{}_{}.png'.format(save_warp_dir, u + 1, v + 1)
        else:
            img_name = '{}/view{}_{}.png'.format(save_OccPred_dir, u + 1, v + 1)
        imageio.imwrite(img_name, rgb_im)