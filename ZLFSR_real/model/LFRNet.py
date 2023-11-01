import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
from model import LF_depth_estimator, Optical_flow_estimator, LF_OccPred
from model import softsplat


class Build_LFRNet(nn.Module):
    def __init__(self, cfg):
        super(Build_LFRNet, self).__init__()

        self.depth_estimator = LF_depth_estimator.LF_depth_estimator(cfg)                                  # LF depth estimator       [B,an2,h,w] --> [B,1,h,w]
        self.bg_flow_estimator = Optical_flow_estimator.Optical_flow_estimator([1,2,4,8,16,32,1,1,1])      # Background Optical_flow_estimator   [B,2,h,w] --> [B,4,h,w]
        self.fg_flow_estimator = Optical_flow_estimator.Optical_flow_estimator([1,2,4,8,16,32,64,1,1])     # Foreground Optical_flow_estimator   [B,2,h,w] --> [B,4,h,w]
        self.LF_OccPredictor = LF_OccPred.LF_OccPred(cfg)                                                  # LF Occlusion prediction   [B*an2,1,h,w] --> [B*an2,1,h,w]

    def forward(self, img_sourceA, img_sourceB):
        # img_source1: LF image      [B,an2,h,w]
        # img_source2: single image  [B,1,h,w]
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

        ################################### LF backward wapring using LF depth ###################################
        # Backward warping all SAIs to central view coordinate
        gridA = []
        for sai_indA in range(ang2):  # row by row
            XA = torch.arange(0, w).view(1, 1, w).expand(Bat, h, w).type_as(img_sourceA)    # [B,h,w]  original image coordinate
            YA = torch.arange(0, h).view(1, h, 1).expand(Bat, h, w).type_as(img_sourceA)    # [B,h,w]
            ind_saiA = torch.arange(ang2)[sai_indA].type_as(img_sourceA)    # 0:an2-1
            ind_sai_wA = ind_saiA % ang                  # X view position
            ind_sai_hA = torch.floor(ind_saiA / ang)     # Y view position
            grid_w_saiA = XA + central_depthA[:, 0, :, :] * (central_x - ind_sai_wA)    # depth value is positive
            grid_h_saiA = YA + central_depthA[:, 0, :, :] * (central_y - ind_sai_hA)
            grid_w_sai_normA = 2.0 * grid_w_saiA / (w-1) - 1.0     # [B,h,w]   Convert to -1:1
            grid_h_sai_normA = 2.0 * grid_h_saiA / (h-1) - 1.0     # [B,h,w]
            grid_saiA = torch.stack((grid_w_sai_normA, grid_h_sai_normA), dim=3)    # [B,h,w,2]
            gridA.append(grid_saiA)
        gridA = torch.cat(gridA, 0)    # [B*an2,h,w,2]
        warp_img_inputA = img_sourceA.view(Bat*ang2, 1, h, w)   # warping input  [B*an2,1,h,w]
        warped_img_stackA = functional.grid_sample(warp_img_inputA, gridA, mode='bilinear').view(Bat, ang2, h, w)    # [B,an2,h,w]
        ###################################################################################################################
        ###################################################################################################################


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

        ############################  Central SAI and single image backward wapring  ##############################
        ##### Backward warping central SAI to viewB // A--->B
        X_flowB = torch.arange(0, w).view(1, 1, w).expand(Bat, h, w).type_as(img_sourceB)     # [B,h,w] original image coordinates
        Y_flowB = torch.arange(0, h).view(1, h, 1).expand(Bat, h, w).type_as(img_sourceB)     # [B,h,w]
        grid_w_flowB = X_flowB + target_flowB[:, 0, :, :] * 100.0        # [B,h,w]  left-->right / target_disp value is positive / horizontal flow
        grid_h_flowB = Y_flowB - target_flowB[:, 1, :, :] * 50.0         # vertical flow
        grid_w_flow_normB = 2.0 * grid_w_flowB / (w-1) - 1.0             # [B,h,w] Convert to -1:1
        grid_h_flow_normB = 2.0 * grid_h_flowB / (h-1) - 1.0             # [B,h,w]
        grid_flowB = torch.stack((grid_w_flow_normB, grid_h_flow_normB), dim=3)     # [B,h,w,2]
        ## central SAI backward warping
        warp_sai_inputA = central_saiA        # [B,1,h,w]
        warped_saiB = functional.grid_sample(warp_sai_inputA, grid_flowB, mode='bilinear')         # [B,1,h,w]
        ## Backward warping // DepthA--->DepthB
        warp_depth_inputA = central_depthA  # [B,1,h,w]
        warped_depthB = functional.grid_sample(warp_depth_inputA, grid_flowB, mode='bilinear')     # [B,1,h,w]

        ##### Backward warping viewB to central SAI // B--->A
        X_flowA = torch.arange(0, w).view(1, 1, w).expand(Bat, h, w).type_as(img_sourceA)     # [B,h,w] original image coordinates
        Y_flowA = torch.arange(0, h).view(1, h, 1).expand(Bat, h, w).type_as(img_sourceA)     # [B,h,w]
        grid_w_flowA = X_flowA - target_flowA[:, 0, :, :] * 100.0        # [B,h,w]  right-->left / target_disp value is positive / horizontal flow
        grid_h_flowA = Y_flowA + target_flowA[:, 1, :, :] * 50.0         # vertical flow
        grid_w_flow_normA = 2.0 * grid_w_flowA / (w-1) - 1.0             # [B,h,w] Convert to -1:1
        grid_h_flow_normA = 2.0 * grid_h_flowA / (h-1) - 1.0             # [B,h,w]
        grid_flowA = torch.stack((grid_w_flow_normA, grid_h_flow_normA), dim=3)    # [B,h,w,2]
        ## Single image backward warping
        warp_sai_inputB = img_sourceB        # [B,1,h,w]
        warped_saiA = functional.grid_sample(warp_sai_inputB, grid_flowA, mode='bilinear')       # [B,1,h,w]
        ###################################################################################################################
        ###################################################################################################################


        #####################################################################################################################
        ##########################################  Generate LFI / forward warping  #########################################
        ## LF forward warping / Generate warped LFI
        novel_lf_imgB = []
        for warp_indB in range(ang2):
            ind_novelB = torch.arange(ang2)[warp_indB].type_as(img_sourceB)    # 0:an2-1
            ind_novel_wB = ind_novelB % ang                   # X view position
            ind_novel_hB = torch.floor(ind_novelB / ang)      # Y view position
            hor_grid_novelB = warped_depthB * (central_x - ind_novel_wB)      # [B,1,h,w]  warped depth is positive
            ver_grid_novelB = warped_depthB * (central_y - ind_novel_hB)
            full_grid_novelB = torch.cat([hor_grid_novelB, ver_grid_novelB], dim=1)     # [B,2,h,w]
            novel_viewB = softsplat.FunctionSoftsplat(tenInput=img_sourceB, tenFlow=full_grid_novelB, tenMetric=None, strType='average')    # [B,1,h,w]
            novel_lf_imgB.append(novel_viewB)
        novel_lf_imgB = torch.cat(novel_lf_imgB, 0)              # [B*an2,1,h,w]
        novel_lf_imgB = novel_lf_imgB.view(Bat, ang2, h, w)      # [B,an2,h,w]

        ####################################### LF Occlusion prediction ###########################################
        OccPred_lf_imgB = self.LF_OccPredictor(novel_lf_imgB.view(Bat*ang2, 1, h, w))     # [B*an2,1,h,w]
        OccPred_lf_imgB = OccPred_lf_imgB.view(Bat, ang2, h, w)                           # [B,an2,h,w]
        ###################################################################################################################
        ###################################################################################################################


        ###################################################################################################################
        ###############################################  Cycle consistency  ###############################################
        ## LF depth estimation
        central_depthB = self.depth_estimator(OccPred_lf_imgB)        # [B,1,h,w]

        ## Depth map backward warping // DepthB--->DepthA
        warp_depth_inputB = central_depthB           # [B,1,h,w]
        warped_depthA = functional.grid_sample(warp_depth_inputB, grid_flowA, mode='bilinear')      # [B,1,h,w]

        ## LF forward warping
        rec_lf_imgA = []
        for warp_indA in range(ang2):
            ind_recA = torch.arange(ang2)[warp_indA].type_as(central_saiA)     # 0:an2-1
            ind_rec_wA = ind_recA % ang                   # X view position
            ind_rec_hA = torch.floor(ind_recA / ang)      # Y view position
            hor_grid_recA = warped_depthA * (central_x - ind_rec_wA)     # [B,1,h,w]  warped depth is positive
            ver_grid_recA = warped_depthA * (central_y - ind_rec_hA)
            full_grid_recA = torch.cat([hor_grid_recA, ver_grid_recA], dim=1)     # [B,2,h,w]
            rec_viewA = softsplat.FunctionSoftsplat(tenInput=central_saiA, tenFlow=full_grid_recA, tenMetric=None, strType='average')    # [B,1,h,w]
            rec_lf_imgA.append(rec_viewA)
        rec_lf_imgA = torch.cat(rec_lf_imgA, 0)             # [B*an2,1,h,w]
        rec_lf_imgA = rec_lf_imgA.view(Bat, ang2, h, w)     # [B,an2,h,w]

        ## LF Occlusion prediction
        OccPred_lf_imgA = self.LF_OccPredictor(rec_lf_imgA.view(Bat*ang2, 1, h, w))     # [B*an2,1,h,w]
        OccPred_lf_imgA = OccPred_lf_imgA.view(Bat, ang2, h, w)                         # [B,an2,h,w]

        return warped_img_stackA, warped_saiA, warped_saiB, target_flowA, target_flowB, \
               central_depthA, central_depthB, \
               novel_lf_imgB, OccPred_lf_imgB, rec_lf_imgA, OccPred_lf_imgA
