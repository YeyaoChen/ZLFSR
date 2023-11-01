import torch
import torch.utils.data as data
import h5py
import numpy as np
import random


##################################################################################
##############################  Read Training data  ##############################
##################################################################################
class TrainSet(data.Dataset):
    def __init__(self, cfg, lfino):
        super(TrainSet, self).__init__()
        self.psize = cfg.patch_size
        self.ang_out = cfg.angular_resolution

        hf = h5py.File(cfg.dataset_path+str(lfino)+'.h5')    # Input path
        self.LFI1 = hf.get('LFI1')                           # color LFI (ycbcr) [ah,aw,h,w,3,N]
        self.LFI1 = self.LFI1[:cfg.angular_resolution, :cfg.angular_resolution, :, :, :, :]
        self.LFI2 = hf.get('LFI2')                           # [ah,aw,h,w,3,N]
        self.LFI2 = self.LFI2[:cfg.angular_resolution, :cfg.angular_resolution, :, :, :, :]

    def __getitem__(self, index):
        lfi1 = self.LFI1[:, :, :, :, :, index]     # [ah,aw,h,w,3]
        lfi1 = lfi1[:, :, :, :, 0]                 # [ah,aw,h,w]    Y channel
        lfi2 = self.LFI2[:, :, :, :, :, index]     # [ah,aw,h,w,3]
        lfi2 = lfi2[:, :, :, :, 0]                 # [ah,aw,h,w]    Y channel

        #############  Crop to patch  #############
        SAI_H, SAI_W = lfi1.shape[2:4]

        x = random.randrange(0, SAI_H-self.psize)
        y = random.randrange(0, SAI_W-self.psize)
        lfi1 = lfi1[:, :, x:x+self.psize, y:y+self.psize]  # [ah,aw,ph,pw]
        lfi2 = lfi2[:, :, x:x+self.psize, y:y+self.psize]  # [ah,aw,ph,pw]

        #############  Brightness augmentation  #############
        lfi1 = lfi1.astype(np.float32)/255.0
        lfi2 = lfi2.astype(np.float32)/255.0
        # if np.random.rand(1) > 0.5:
        #     factor = random.uniform(0.9, 1.1)
        #     lfi1 = np.power(lfi1, factor)
        #     lfi2 = np.power(lfi2, factor)

        ############  Get input index  #############
        ind_all = np.arange(self.ang_out * self.ang_out).reshape(self.ang_out, self.ang_out)
        ind_central = ind_all[self.ang_out//2, self.ang_out//2]
        ind_central = ind_central.reshape(-1)    # central SAI index
            
        ############  Get input and label  #############
        lfi1 = lfi1.reshape(-1, self.psize, self.psize)     # [ah2,ph,pw]
        lfi2 = lfi2.reshape(-1, self.psize, self.psize)     # [ah2,ph,pw]
        single_img = lfi2[ind_central, :, :]                # [1,ph,pw]

        #############  Convert to tensor  #############
        input_lf = torch.from_numpy(lfi1)               # [an2,h,w]
        input_single = torch.from_numpy(single_img)     # [1,h,w]
        return input_lf, input_single

    def __len__(self):
        return self.LFI1.shape[5]


##############################################################################
##############################  Read Test data  ##############################
##############################################################################
class TestSet(data.Dataset):
    def __init__(self, cfg, lfno):
        super(TestSet, self).__init__()
        self.ang_out = cfg.angular_resolution

        hf = h5py.File(cfg.dataset_path+str(lfno)+'.h5')    # Input path
        self.LFI_ycbcr1 = hf.get('LFI1')                    # [ah,aw,h,w,3,N]
        self.LFI_ycbcr1 = self.LFI_ycbcr1[:cfg.angular_resolution, :cfg.angular_resolution, :, :, :, :]
        self.LFI_ycbcr2 = hf.get('LFI2')                    # [ah,aw,h,w,3,N]
        self.LFI_ycbcr2 = self.LFI_ycbcr2[:cfg.angular_resolution, :cfg.angular_resolution, :, :, :, :]

    def __getitem__(self, index):
        view_H, view_W = self.LFI_ycbcr1.shape[2:4]

        lfi_ycbcr1 = self.LFI_ycbcr1[:, :, :, :, :, index]           # [ah,aw,h,w,3]
        lfi_ycbcr1 = lfi_ycbcr1.reshape(-1, view_H, view_W, 3)       # [an2,h,w,3]
        lfi_ycbcr2 = self.LFI_ycbcr2[:, :, :, :, :, index]           # [ah,aw,h,w,3]
        lfi_ycbcr2 = lfi_ycbcr2.reshape(-1, view_H, view_W, 3)       # [an2,h,w,3]

        ################  Input  ################
        ind_all = np.arange(self.ang_out * self.ang_out).reshape(self.ang_out, self.ang_out)
        ind_source = ind_all[self.ang_out//2, self.ang_out//2]
        ind_source = ind_source.reshape(-1)     # central view index

        input1 = lfi_ycbcr1[:, :, :, 0]              # [an2,h,w]
        input2 = lfi_ycbcr2[ind_source, :, :, 0]     # [1,h,w]

        #############  Ground Truth  #############
        target_y = lfi_ycbcr2[:, :, :, 0]            # [an2,h,w]

        input1 = torch.from_numpy(input1.astype(np.float32)/255.0)
        input2 = torch.from_numpy(input2.astype(np.float32)/255.0)
        target_y = torch.from_numpy(target_y.astype(np.float32)/255.0)

        #############  Use GT for visual results  #############
        lfi_ycbcr = torch.from_numpy(lfi_ycbcr2.astype(np.float32)/255.0)
        return input1, input2, target_y, lfi_ycbcr

    def __len__(self):
        return self.LFI_ycbcr1.shape[5]