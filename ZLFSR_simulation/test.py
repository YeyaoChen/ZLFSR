import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from datetime import datetime
import pandas as pd
import scipy.io as scio
import imageio
from load_dataset import TestSet
from model import LFRNet
from utils import calculate_total_scores, save_LFimg


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU setting
#######################################################################################
# Test parameter settings
parser = argparse.ArgumentParser(description="Heterogeneous light field reconstruction -- test mode")
parser.add_argument("--angular_resolution", type=int, default=7, help="Angular number of the target light field")
parser.add_argument("--model_path", type=str, default="trained_model_LR/scene", help="Pretrained model path")
parser.add_argument("--dataset_path", type=str, default="matlab/dataset_LR/data", help="Testing data path")
cfg = parser.parse_args()
print(cfg)

#######################################################################################
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#######################################################################################

for LF_num in range(1,2):
    # Model path
    pretrained_model_path = cfg.model_path + str(LF_num) + '/model_epoch_300.pth'
    if not os.path.exists(pretrained_model_path):
        print('Pretrained model folder is not found ')

    # Evaluation results save path
    if not os.path.exists('evaluation_results'):
        os.makedirs('evaluation_results')

    #######################################################################################
    # Data loader
    print('===> Loading test dataset_' + str(LF_num))
    test_set = TestSet(cfg, LF_num)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
    print('Loaded {} LFI from {}{}.h5'.format(len(test_loader), cfg.dataset_path, str(LF_num)))

    #######################################################################################
    # Build model
    print("Building LFRNet")
    model_test = LFRNet.Build_LFRNet(cfg).to(device)

    #######################################################################################
    total = sum([param.nelement() for param in model_test.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    #######################################################################################
    #######################################################################################
    def test():
        lf_list = []
        lf_psnr1_list = []
        lf_ssim1_list = []
        lf_psnr2_list = []
        lf_ssim2_list = []

        csv_name1 = 'evaluation_results/Warp.csv'
        csv_name2 = 'evaluation_results/OccPred.csv'
        ang2 = cfg.angular_resolution ** 2
        center_index = ang2 // 2

        model_test.eval()
        with torch.no_grad():   # Test
            for k, batch in enumerate(test_loader):
                start_time = datetime.now()
                print('testing LF{} batch {}'.format(LF_num, k))
                ######################### Light Field Reconstruction #########################
                input1, input2, target_y, lfi_ycbcr = batch[0], batch[1], batch[2].numpy(), batch[3].numpy()

                ############### forward inference ###############
                input1 = input1.to(device)    # input LF image       [b,an2,h,w]
                input2 = input2.to(device)    # input single image   [b,1,h,w]

                gt_sai1 = input1[:, center_index, :, :].unsqueeze(1)
                gt_sai2 = input2

                bg_mask_map, fg_mask_map, warped_img_stack1, warped_sai1, warped_sai2, infer_flow1, infer_flow2, \
                infer_depth1, infer_depth2, warped_depth1, warped_depth2, \
                novel_lf_img2, OccPred_lf_img2, rec_lf_img1, OccPred_lf_img1 = model_test(input1, input2)

                elapsed_time = datetime.now() - start_time
                print('Elapsed time: %s' % (elapsed_time))

                ####################### Calculate PSNR/SSIM for each SAI #######################
                warp_y = novel_lf_img2.cpu().numpy()          # [B,an2,h,w]
                OccPred_y = OccPred_lf_img2.cpu().numpy()     # [B,an2,h,w]
                lf_psnr1, lf_ssim1 = calculate_total_scores(warp_y, target_y, csv_name1, LF_num, ang2)
                lf_psnr2, lf_ssim2 = calculate_total_scores(OccPred_y, target_y, csv_name2, LF_num, ang2)
                print('PSNR1: {:.2f}, SSIM1: {:.4f}'.format(lf_psnr1, lf_ssim1))
                print('PSNR2: {:.2f}, SSIM2: {:.4f}'.format(lf_psnr2, lf_ssim2))

                ####################### Calculate PSNR/SSIM for each LFI #######################
                lf_list.append(LF_num)
                lf_psnr1_list.append(lf_psnr1)
                lf_ssim1_list.append(lf_ssim1)
                lf_psnr2_list.append(lf_psnr2)
                lf_ssim2_list.append(lf_ssim2)

                ####################### Save LF images and depth maps #######################
                gt_ycbcr = lfi_ycbcr       # GT LFI [B,an2,h,w,3]
                save_LFimg(warp_y, gt_ycbcr, 'saveImg/Warp/', LF_num, cfg.angular_resolution, isWarp=True)
                save_LFimg(OccPred_y, gt_ycbcr, 'saveImg/OccPred/', LF_num, cfg.angular_resolution, isWarp=False)
                scio.savemat('saveImg/Warp/' + str(LF_num) + '/depth1.mat', {'Depth1': infer_depth1.cpu().numpy().squeeze()})                 # [h,w]
                scio.savemat('saveImg/Warp/' + str(LF_num) + '/flow1.mat', {'Flow1': infer_flow1.cpu().numpy().squeeze().transpose(1,2,0)})   # [h,w,2]
                scio.savemat('saveImg/Warp/' + str(LF_num) + '/warped_depth1.mat', {'Warped_depth1': warped_depth1.cpu().numpy().squeeze()})  # [h,w]

                scio.savemat('saveImg/Warp/' + str(LF_num) + '/depth2.mat', {'Depth2': infer_depth2.cpu().numpy().squeeze()})
                scio.savemat('saveImg/Warp/' + str(LF_num) + '/flow2.mat', {'Flow2': infer_flow2.cpu().numpy().squeeze().transpose(1,2,0)})
                scio.savemat('saveImg/Warp/' + str(LF_num) + '/warped_depth2.mat', {'Warped_depth2': warped_depth2.cpu().numpy().squeeze()})

                # save *mat file
                _, _, spa1, spa2 = warp_y.shape
                warp_y = warp_y.squeeze(0).reshape(cfg.angular_resolution, cfg.angular_resolution, spa1, spa2)        # [an,an,h,w]
                OccPred_y = OccPred_y.squeeze(0).reshape(cfg.angular_resolution, cfg.angular_resolution, spa1, spa2)  # [an,an,h,w]
                scio.savemat('saveImg/Warp/' + str(LF_num) + '/warp.mat', {'PredLF1': warp_y})
                scio.savemat('saveImg/OccPred/' + str(LF_num) + '/OccPred.mat', {'PredLF2': OccPred_y})

                for ss in range(49):
                    warped = (warped_img_stack1[0, ss].cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8)
                    u = ss // 7
                    v = ss % 7
                    imageio.imwrite('saveImg/Warp/'+str(LF_num)+'/'+str(u+1)+'_'+str(v+1)+'.png', warped)

                warped_sai1 = (warped_sai1[0, 0].cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8)
                imageio.imwrite('saveImg/Warp/'+str(LF_num)+'/L.png', warped_sai1)
                warped_sai2 = (warped_sai2[0, 0].cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8)
                imageio.imwrite('saveImg/Warp/' + str(LF_num) + '/R.png', warped_sai2)

                gt_sai1 = (gt_sai1[0, 0].cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8)
                imageio.imwrite('saveImg/Warp/' + str(LF_num) + '/GT_L.png', gt_sai1)
                gt_sai2 = (gt_sai2[0, 0].cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8)
                imageio.imwrite('saveImg/Warp/' + str(LF_num) + '/GT_R.png', gt_sai2)

                imageio.imwrite('saveImg/Warp/' + str(LF_num) + '/bg.png', (bg_mask_map[0, 0].cpu().numpy()*255.0).astype(np.uint8))
                imageio.imwrite('saveImg/Warp/' + str(LF_num) + '/fg.png', (fg_mask_map[0, 0].cpu().numpy()*255.0).astype(np.uint8))

            dataframe1_lfi = pd.DataFrame({'LFI': lf_list, 'Warp PSNR Y': lf_psnr1_list, 'Warp SSIM Y': lf_ssim1_list})
            dataframe1_lfi.to_csv(csv_name1, index=False, sep=',', mode='a')
            dataframe2_lfi = pd.DataFrame({'LFI': lf_list, 'OccPred PSNR Y': lf_psnr2_list, 'OccPred SSIM Y': lf_ssim2_list})
            dataframe2_lfi.to_csv(csv_name2, index=False, sep=',', mode='a')

    #######################################################################################
    print('===> test')
    checkpoints = torch.load(pretrained_model_path)
    ckp_dict = checkpoints['model']

    # load the trained model parameters
    print('loaded model from ' + pretrained_model_path)
    model_test_dict = model_test.state_dict()
    ckp_dict_refine = {k: v for k, v in ckp_dict.items() if k in model_test_dict}
    model_test_dict.update(ckp_dict_refine)
    model_test.load_state_dict(model_test_dict)
    # Begin testing
    test()


