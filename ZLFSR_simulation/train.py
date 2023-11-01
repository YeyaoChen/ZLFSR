import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from os.path import join
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import imageio
from load_dataset import TrainSet
from model import LFRNet
from loss import get_loss


os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # GPU setting
#######################################################################################
# Training parameter settings
parser = argparse.ArgumentParser(description="Heterogeneous light field reconstruction -- train mode")
parser.add_argument("--angular_resolution", type=int, default=7, help="Angular number of the target light field")
parser.add_argument("--dataset_path", type=str, default="matlab/dataset_LR/data", help="Training data path")
parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
parser.add_argument("--patch_size", type=int, default=240, help="Training patch size")
parser.add_argument("--train_epoch", type=int, default=300, help="Training epoch")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--step", type=int, default=200, help="Learning rate decay every n epochs")
parser.add_argument("--gamma", type=float, default=0.5, help="Learning rate decay")
parser.add_argument('--l1_weight', type=float, default=1.0, help='The weight of L1 reconstruction loss')
parser.add_argument('--ssim_weight', type=float, default=0.5, help='The weight of ssim loss')
parser.add_argument('--perceptual_weight', type=float, default=0, help='The weight of perceptual loss')
parser.add_argument('--smooth_weight', type=float, default=0.5, help='The weight of smooth loss')
parser.add_argument('--epigrad_weight', type=float, default=0.5, help='The weight of EPI gradient loss')
parser.add_argument('--detail_weight', type=float, default=0, help='The weight of detail loss')
parser.add_argument("--resume_epoch", type=int, default=0, help="Resume from checkpoint epoch")
parser.add_argument("--num_cp", type=int, default=10, help="Number of epochs for saving checkpoint")
parser.add_argument("--num_snapshot", type=int, default=1, help="Number of epochs for saving loss figure")
parser.add_argument("--Is_buffer", type=int, default=0, help="Save the image in training")
cfg = parser.parse_args()
print(cfg)

#######################################################################################
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#######################################################################################
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

# Weight initialization
def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)

# Loss functions
loss_all = get_loss(cfg, device)
print(loss_all)

#######################################################################################
for LF_num in range(1,2):
    # model save path
    model_dir = 'trained_model_LR/scene' + str(LF_num)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Training save image
    if cfg.Is_buffer:
        train_visual_dir = 'training_buffer_LR/scene' + str(LF_num)
        if not os.path.exists(train_visual_dir):
            os.makedirs(train_visual_dir)

    #######################################################################################
    # Data loader
    print('===> Loading train dataset_' + str(LF_num))
    train_set = TrainSet(cfg, LF_num)
    train_loader = DataLoader(dataset=train_set, batch_size=cfg.batch_size, shuffle=True)
    print('Loaded {} LFI from {}{}.h5'.format(len(train_loader), cfg.dataset_path, str(LF_num)))
    print('train data number:', len(train_loader))

    #######################################################################################
    # Build model
    print("Building LFRNet")
    model_train = LFRNet.Build_LFRNet(cfg).to(device)
    # Initialize weight
    model_train.apply(weights_init_xavier)
    for para_name in model_train.state_dict():  # print trained parameters
        print(para_name)

    #######################################################################################
    total = sum([param.nelement() for param in model_train.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    #######################################################################################
    # Optimizer and loss logger
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_train.parameters()), lr=cfg.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step, gamma=cfg.gamma)
    losslogger = defaultdict(list)

    #######################################################################################
    # Reload previous parameters
    if cfg.resume_epoch:
        resume_path = join(model_dir, 'model_epoch_{}.pth'.format(cfg.resume_epoch))
        if os.path.isfile(resume_path):
            print("==>Loading model parameters 'epoch{}'".format(resume_path))
            checkpoints = torch.load(resume_path)
            model_train.load_state_dict(checkpoints['model'])
            optimizer.load_state_dict(checkpoints['optimizer'])
            scheduler.load_state_dict(checkpoints['scheduler'])
            losslogger = checkpoints['losslogger']
        else:
            print("==> no model found at 'epoch{}'".format(cfg.resume_epoch))

    #######################################################################################
    #######################################################################################
    def train(epoch, start_time):
        # Train
        model_train.train()
        scheduler.step()
        loss_epoch = 0.      # Total loss per epoch
        an_2 = cfg.angular_resolution ** 2
        center_index = an_2 // 2

        print('Current epoch learning rate: %e' % (optimizer.state_dict()['param_groups'][0]['lr']))
        for tp in range(15):
            for tq, batch in enumerate(train_loader):
                in_lfi, in_single_img = batch[0].to(device), batch[1].to(device)
                label_center = in_lfi[:, center_index, :, :].unsqueeze(1)                                 # [b,1,h,w]
                label_center_stack = in_lfi[:, center_index, :, :].unsqueeze(1).repeat(1, an_2, 1, 1)     # [b,an2,h,w]

                ##################### Forward inference #####################
                _, _, warp_center_stack, warp_single_saiA, warp_single_saiB, pred_flowA, pred_flowB, \
                pred_depthA, pred_depthB, pred_warp_depthA, pred_warp_depthB, \
                pred_warp_lfi, pred_occ_lfi, pred_warp_lfi_cycle, pred_occ_lfi_cycle = model_train(in_lfi, in_single_img)

                ###################################### Total loss ######################################
                # LF depth loss
                depth_warp_l1_loss = cfg.l1_weight * loss_all['l1_loss'](warp_center_stack, label_center_stack)
                depth_warp_ssim_loss = cfg.ssim_weight * loss_all['ssim_loss'](warp_center_stack, label_center_stack)
                depth_map_smooth_loss = cfg.smooth_weight * loss_all['smooth_loss'](pred_depthA, label_center)
                depth_loss = depth_warp_l1_loss + depth_warp_ssim_loss + depth_map_smooth_loss


                # Optical flow loss
                flow_warp_l1_loss = cfg.l1_weight * loss_all['l1_loss'](warp_single_saiA, label_center) + \
                                    cfg.l1_weight * loss_all['l1_loss'](warp_single_saiB, in_single_img)
                flow_warp_ssim_loss = cfg.ssim_weight * loss_all['ssim_loss'](warp_single_saiA, label_center) +  \
                                      cfg.ssim_weight * loss_all['ssim_loss'](warp_single_saiB, in_single_img)
                flow_map_smooth_loss = cfg.smooth_weight * loss_all['smooth_loss'](pred_flowA, label_center.repeat(1, 2, 1, 1)) + \
                                       cfg.smooth_weight * loss_all['smooth_loss'](pred_flowB, in_single_img.repeat(1, 2, 1, 1))
                flow_loss = flow_warp_l1_loss + flow_warp_ssim_loss + flow_map_smooth_loss


                # Cycle consistency loss
                cycle_l1_loss_all = cfg.l1_weight * loss_all['l1_loss'](pred_warp_lfi_cycle, in_lfi) + \
                                    cfg.l1_weight * loss_all['l1_loss'](pred_occ_lfi_cycle, in_lfi)
                cycle_ssim_loss_all = cfg.ssim_weight * loss_all['ssim_loss'](pred_warp_lfi_cycle, in_lfi) + \
                                      cfg.ssim_weight * loss_all['ssim_loss'](pred_occ_lfi_cycle, in_lfi)
                cycle_epi_loss_all = cfg.epigrad_weight * loss_all['epigrad_loss'](pred_warp_lfi_cycle, in_lfi) + \
                                     cfg.epigrad_weight * loss_all['epigrad_loss'](pred_occ_lfi_cycle, in_lfi)
                cycle_loss_all = cycle_l1_loss_all + cycle_ssim_loss_all + cycle_epi_loss_all

                # Total loss
                loss = depth_loss + flow_loss * 0.5 + cycle_loss_all

                # Cumulative loss
                loss_epoch += loss.item()

                ####################### Backward and optimize #######################
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            ##################### Save training results #####################
            if cfg.Is_buffer:
                if epoch % cfg.num_cp == 0 and tp % 15 == 0:
                    in_name1 = '{}/in_view0_epoch{}_k{}.png'.format(train_visual_dir, epoch, tp)
                    in_name2 = '{}/in_single_epoch{}_k{}.png'.format(train_visual_dir, epoch, tp)
                    warp_name = '{}/warp_view0_epoch{}_k{}.png'.format(train_visual_dir, epoch, tp)
                    refine_name = '{}/refine_view0_epoch{}_k{}.png'.format(train_visual_dir, epoch, tp)

                    save_in1 = (in_lfi[0, 0, :, :].detach().cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8)
                    save_in2 = (in_single_img[0, 0, :, :].detach().cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8)
                    save_warp = (pred_warp_lfi[0, 0, :, :].detach().cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8)
                    save_refine = (pred_occ_lfi[0, 0, :, :].detach().cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8)

                    imageio.imwrite(in_name1, save_in1)
                    imageio.imwrite(in_name2, save_in2)
                    imageio.imwrite(warp_name, save_warp)
                    imageio.imwrite(refine_name, save_refine)

        ############################## Print results ##############################
        # print('Training==>>Epoch: %d,  depth loss: %s,  disp loss: %s,  cycle loss: %s,  total loss: %s'
        #       % (epoch, depth_loss.item(), disp_loss.item(), cycle_loss_all.item(), loss.item()))
        # print('Training==>>Epoch: %d,  depth l1 loss: %s,  depth ssim loss: %s,  depth smooth loss: %s,  depth loss: %s'
        #       % (epoch, depth_warp_l1_loss.item(), depth_warp_ssim_loss.item(), depth_map_smooth_loss.item(), depth_loss.item()))

        # print('Training==>>Epoch: %d,  flow l1 loss: %s,  flow ssim loss: %s,  flow smooth loss: %s,  flow loss: %s'
        #       % (epoch, flow_warp_l1_loss.item(), flow_warp_ssim_loss.item(), flow_map_smooth_loss.item(), flow_loss.item()))

        # print('Training==>>Epoch: %d,  cycle l1 loss: %s,  cycle ssim loss: %s, cycle EPI loss: %s,  cycle loss: %s'
        #     % (epoch, cycle_l1_loss_all.item(), cycle_ssim_loss_all.item(), cycle_epi_loss_all.item(), cycle_loss_all.item()))


        losslogger['epoch'].append(epoch)
        losslogger['loss'].append(loss_epoch/len(train_loader))
        elapsed_time = datetime.now() - start_time
        print('Training==>>Dataset: %d, Epoch: %d,  loss: %s,  elapsed time: %s' % (LF_num, epoch, loss_epoch/len(train_loader), elapsed_time))

    #######################################################################################
    print('==>training')
    start_time = datetime.now()
    for epoch in range(cfg.resume_epoch+1, cfg.train_epoch+1):
        train(epoch, start_time)

        # save trained model parameters
        if epoch % cfg.num_cp == 0:
            model_save_path = join(model_dir, "model_epoch_{}.pth".format(epoch))
            state = {'epoch': epoch, 'model': model_train.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'losslogger': losslogger}
            torch.save(state, model_save_path)
            print("checkpoint saved to {}".format(model_save_path))

        # save loss figure
        if epoch % cfg.num_snapshot == 0:
            plt.figure()
            plt.title('loss')
            plt.plot(losslogger['epoch'], losslogger['loss'])
            plt.savefig(model_dir + "/loss.png")
            plt.close('all')


