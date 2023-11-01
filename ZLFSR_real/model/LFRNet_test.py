import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
from model import LF_depth_estimator, Optical_flow_estimator, LF_OccPred
from model import softsplat


class Build_LFRNet_test(nn.Module):
    def __init__(self, cfg):
        super(Build_LFRNet_test, self).__init__()

        self.depth_estimator = LF_depth_estimator.LF_depth_estimator(cfg)                                  # LF depth estimator       [B,an2,h,w] --> [B,1,h,w]
        self.bg_flow_estimator = Optical_flow_estimator.Optical_flow_estimator([1,2,4,8,16,32,1,1,1])      # Background Optical_flow_estimator   [B,2,h,w] --> [B,4,h,w]
        self.fg_flow_estimator = Optical_flow_estimator.Optical_flow_estimator([1,2,4,8,16,32,64,1,1])     # Foreground Optical_flow_estimator   [B,2,h,w] --> [B,4,h,w]
        self.LF_OccPredictor = LF_OccPred.LF_OccPred(cfg)                                                  # LF Occlusion prediction   [B*an2,1,h,w] --> [B*an2,1,h,w]

    def forward(self, img_sourceA, img_sourceB, img_cb, img_cr):
        # img_source1: LF image          [B,an2,h,w]
        # img_source2: single image      [B,1,h,w]
        # img_cbcr: single image (CbCr)  [B,1,h,w,2]
        Bat, ang2, h, w = img_sourceA.shape    # [B,an2,h,w]
        ang = int(np.sqrt(ang2))   # angular resolution
        central_x = ang // 2       # 7-->3 horizontal direction
        central_y = ang // 2       # 7-->3 vertical direction
        central_ind = ang2 // 2    # 49-->24
        central_saiA = img_sourceA[:, central_ind, :, :].unsqueeze(1)     # central SAI [B,1,h,w]

        ###################################################################################################################
        ##############################################  LF depth estimation  ##############################################
        # Thanks to the rigid structure of LF, the horizontal and vertical parallax are the same
        central_depthA = self.depth_estimator(img_sourceA)       # [B,1,h,w]

        ###################################################################################################################
        #############################################  Optical_flow estimation  ############################################
        # Due to the inaccurate calibration of LF camera and 2D camera, there is horizontal and vertical optical flow
        # Background and foreground mask
        norm_depthA = (central_depthA - central_depthA.min()) / (central_depthA.max() - central_depthA.min())    # [B,1,h,w]
        one_mask = torch.ones(norm_depthA.shape).type_as(norm_depthA)       # [B,1,h,w]
        zero_mask = torch.zeros(norm_depthA.shape).type_as(norm_depthA)
        bg_mask = torch.where(norm_depthA < 0.6, one_mask, zero_mask)       # [B,1,h,w]
        fg_mask = one_mask - bg_mask

        # Background flow estimation
        flow_input = torch.cat([central_saiA, img_sourceB], dim=1)            # [B,2,h,w]
        bg_target_flow = self.bg_flow_estimator(flow_input * (bg_mask.repeat(1, 2, 1, 1)))
        bg_target_flowA = bg_target_flow[:, :2, :, :]     # [B,2,h,w]
        bg_target_flowB = bg_target_flow[:, 2:, :, :]     # [B,2,h,w]
        # Foreground flow estimation
        fg_target_flow = self.fg_flow_estimator(flow_input * (fg_mask.repeat(1, 2, 1, 1)))
        fg_target_flowA = fg_target_flow[:, :2, :, :]     # [B,2,h,w]
        fg_target_flowB = fg_target_flow[:, 2:, :, :]     # [B,2,h,w]

        target_flowA = bg_target_flowA * (bg_mask.repeat(1, 2, 1, 1)) + fg_target_flowA * (fg_mask.repeat(1, 2, 1, 1))
        target_flowB = bg_target_flowB * (bg_mask.repeat(1, 2, 1, 1)) + fg_target_flowB * (fg_mask.repeat(1, 2, 1, 1))

        ############################  Central SAI and single image backward warping  ##############################
        ##### Backward warping central SAI to viewB // A--->B
        X_flowB = torch.arange(0, w).view(1, 1, w).expand(Bat, h, w).type_as(img_sourceB)     # [B,h,w] original image coordinates
        Y_flowB = torch.arange(0, h).view(1, h, 1).expand(Bat, h, w).type_as(img_sourceB)     # [B,h,w]
        grid_w_flowB = X_flowB + target_flowB[:, 0, :, :] * 100.0        # [B,h,w]  left-->right / target_disp value is positive / horizontal flow
        grid_h_flowB = Y_flowB - target_flowB[:, 1, :, :] * 50.0         # vertical flow
        grid_w_flow_normB = 2.0 * grid_w_flowB / (w-1) - 1.0             # [B,h,w] Convert to -1:1
        grid_h_flow_normB = 2.0 * grid_h_flowB / (h-1) - 1.0             # [B,h,w]
        grid_flowB = torch.stack((grid_w_flow_normB, grid_h_flow_normB), dim=3)     # [B,h,w,2]
        ## Backward warping // DepthA--->DepthB
        warp_depth_inputA = central_depthA  # [B,1,h,w]
        warped_depthB = functional.grid_sample(warp_depth_inputA, grid_flowB, mode='bilinear')     # [B,1,h,w]


        #####################################################################################################################
        ##########################################  Generate LFI / forward warping  #########################################
        ## LF forward warping / Generate warped LFI
        novel_lf_imgB = []
        novel_lf_cb = []
        novel_lf_cr = []

        for warp_indB in range(ang2):
            ind_novelB = torch.arange(ang2)[warp_indB].type_as(img_sourceB)    # 0:an2-1
            ind_novel_wB = ind_novelB % ang                   # X view position
            ind_novel_hB = torch.floor(ind_novelB / ang)      # Y view position
            hor_grid_novelB = warped_depthB * (central_x - ind_novel_wB)      # [B,1,h,w]  warped depth is positive
            ver_grid_novelB = warped_depthB * (central_y - ind_novel_hB)
            full_grid_novelB = torch.cat([hor_grid_novelB, ver_grid_novelB], dim=1)     # [B,2,h,w]
            novel_viewB = softsplat.FunctionSoftsplat(tenInput=img_sourceB, tenFlow=full_grid_novelB, tenMetric=None, strType='average')    # [B,1,h,w]
            novel_lf_imgB.append(novel_viewB)
            novel_cb = softsplat.FunctionSoftsplat(tenInput=img_cb, tenFlow=full_grid_novelB, tenMetric=None, strType='average')    # [B,1,h,w]
            novel_lf_cb.append(novel_cb)
            novel_cr = softsplat.FunctionSoftsplat(tenInput=img_cr, tenFlow=full_grid_novelB, tenMetric=None, strType='average')    # [B,1,h,w]
            novel_lf_cr.append(novel_cr)

        novel_lf_imgB = torch.cat(novel_lf_imgB, 0)              # [B*an2,1,h,w]
        novel_lf_imgB = novel_lf_imgB.view(Bat, ang2, h, w)      # [B,an2,h,w]
        novel_lf_cb = torch.cat(novel_lf_cb, 0)                   # [B*an2,1,h,w]
        novel_lf_cb = novel_lf_cb.view(Bat, ang2, h, w)           # [B,an2,h,w]
        novel_lf_cr = torch.cat(novel_lf_cr, 0)                    # [B*an2,1,h,w]
        novel_lf_cr = novel_lf_cr.view(Bat, ang2, h, w)            # [B,an2,h,w]
        novel_lf_img = torch.cat([novel_lf_imgB.unsqueeze(-1), novel_lf_cb.unsqueeze(-1), novel_lf_cr.unsqueeze(-1)], -1)

        ####################################### LF Occlusion prediction ###########################################
        OccPred_lf_imgB = self.LF_OccPredictor(novel_lf_imgB.view(Bat*ang2, 1, h, w))     # [B*an2,1,h,w]
        OccPred_lf_imgB = OccPred_lf_imgB.view(Bat, ang2, h, w)                           # [B,an2,h,w]
        OccPred_lf_img = torch.cat([OccPred_lf_imgB.unsqueeze(-1), novel_lf_cb.unsqueeze(-1), novel_lf_cr.unsqueeze(-1)], -1)

        return target_flowA, target_flowB, central_depthA, novel_lf_img, OccPred_lf_img
