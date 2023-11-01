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
from model import LFRNet_test
from utils import calculate_total_scores, save_LFimg


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU setting
#######################################################################################
# Test parameter settings
parser = argparse.ArgumentParser(description="Heterogeneous light field reconstruction -- test mode")
parser.add_argument("--angular_resolution", type=int, default=7, help="Angular number of the target light field")
parser.add_argument("--model_path", type=str, default="checkpoints/scene", help="Pre_trained model path")
parser.add_argument("--dataset_path", type=str, default="matlab/RealData/data", help="Testing data path")
cfg = parser.parse_args()
print(cfg)

#######################################################################################
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#######################################################################################

for LF_num in range(1,2):
    # Model path
    pretrained_model_path = cfg.model_path + str(LF_num) + '/Trained_model.pth'
    if not os.path.exists(pretrained_model_path):
        print('Pretrained model folder is not found ')

    #######################################################################################
    # Data loader
    print('===> Loading test dataset_' + str(LF_num))
    test_set = TestSet(cfg, LF_num)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
    print('Loaded {} LFI from {}{}.h5'.format(len(test_loader), cfg.dataset_path, str(LF_num)))

    #######################################################################################
    # Build model
    print("Building LFRNet")
    model_test = LFRNet_test.Build_LFRNet_test(cfg).to(device)

    #######################################################################################
    #######################################################################################
    def test():

        model_test.eval()
        with torch.no_grad():   # Test
            for k, batch in enumerate(test_loader):
                start_time = datetime.now()
                print('testing LF{} batch {}'.format(LF_num, k))
                ######################### Light Field Reconstruction #########################
                input1, input2, input_cb2, input_cr2 = batch[0], batch[1], batch[2], batch[3]
                input1 = input1.to(device)    # input LF image       [b,an2,h,w]
                input2 = input2.to(device)    # input single image   [b,1,h,w]
                input_cb2 = input_cb2.to(device)  # input single image   [b,1,h,w]
                input_cr2 = input_cr2.to(device)  # input single image   [b,1,h,w]

                ############### Forward inference ###############
                _, _, _, novel_lf_img2, OccPred_lf_img2 = model_test(input1, input2, input_cb2, input_cr2)

                elapsed_time = datetime.now() - start_time
                print('Elapsed time: %s' % (elapsed_time))

                ####################### Save LFI #######################
                warp_ycbcr = novel_lf_img2.cpu().numpy()           # [B,an2,h,w,3]
                OccPred_ycbcr = OccPred_lf_img2.cpu().numpy()      # [B,an2,h,w,3]

                save_LFimg(warp_ycbcr, 'outputs/Warp/', LF_num, cfg.angular_resolution, isWarp=True)
                save_LFimg(OccPred_ycbcr, 'outputs/OccPred/', LF_num, cfg.angular_resolution, isWarp=False)

                # save *mat file
                _, _, spa1, spa2, col = warp_ycbcr.shape
                warp_ycbcr = warp_ycbcr.squeeze(0).reshape(cfg.angular_resolution, cfg.angular_resolution, spa1, spa2, col)        # [an,an,h,w]
                OccPred_ycbcr = OccPred_ycbcr.squeeze(0).reshape(cfg.angular_resolution, cfg.angular_resolution, spa1, spa2, col)  # [an,an,h,w]
                scio.savemat('outputs/Warp/' + str(LF_num) + '/warp.mat', {'PredLF1': warp_ycbcr})
                scio.savemat('outputs/OccPred/' + str(LF_num) + '/OccPred.mat', {'PredLF2': OccPred_ycbcr})

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


