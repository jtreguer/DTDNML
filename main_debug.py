#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Before run this file, please ensure running <python -m visdom.server> in current environment.
Then, please go to http:localhost://#display_port# to see the visulizations.

Training time full time 524363 sec, 146 hours
"""

import torch
import time
import hues
import os
from data import get_dataloader
from model import create_model
from options.train_options import TrainOptions
from utils.visualizer import Visualizer
from utils.util import load_checkpoint, save_checkpoint
import scipy.io as sio
import numpy as np
import random
from tqdm import tqdm


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
setup_seed(5)

if __name__ == "__main__":

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    start_time = time.time()

    train_opt = TrainOptions().parse()

    train_opt.niter = 900
    train_opt.niter_decay = 2100
    train_opt.lr = 1e-2
    train_opt.lr_decay_iters = 300
    train_opt.display_port = 8097
    
    """KSC"""
    # train_opt.name = 'ksc_scale_8'
    # train_opt.data_name = "ksc"
    # train_opt.srf_name = "ksc"  # 'Landsat8_BGR'
    # train_opt.mat_name = "KSC"

    """Sandiego"""
    train_opt.name = 'sandiego_scale_4' #'sandiego_scale_8'
    train_opt.data_path_name = "sandiego"
    train_opt.data_img_name = "sandiego_ort"
    train_opt.srf_name = "Landsat8_BGRI_SRF"  # 'Landsat8_BGR'
    train_opt.start_x_pixel = 800
    train_opt.start_y_pixel = 200
    train_opt.window_size = 64
    
    """chikusei"""
    # train_opt.name = 'chikusei_scale_8'
    # train_opt.data_name = "chikusei"
    # train_opt.srf_name = "chikusei"  # 'Landsat8_BGR'
    # train_opt.mat_name = "chikusei"

    """Indian Pines"""
    # train_opt.name = 'indian_scale_4'
    # train_opt.data_name = "indian"
    # train_opt.srf_name = "indian"  # 'Landsat8_BGR'
    # train_opt.mat_name = "indian"

    """WaDC"""
    # train_opt.name = 'wadc_scale_4'
    # train_opt.data_name = 'wadc'
    # train_opt.srf_name = 'wadc'  # 'Landsat8_BGR'
    # train_opt.mat_name = 'WaDC3'
    
    """Pavia"""
    # train_opt.name = 'pavia_scale_8'
    # train_opt.data_name = 'pavia_cat'
    # train_opt.srf_name = 'paviac'  # 'Landsat8_BGR'
    # train_opt.mat_name = 'PaviaC'

    """CAVE"""
    # train_opt.name = 'CAVE_04'
    # train_opt.data_name = 'CAVE'
    # train_opt.srf_name = 'Nikon_D700_Qu'  # 'Landsat8_BGR'
    # # train_opt.mat_name = 'chart_and_stuffed_toy_ms'
    # # train_opt.mat_name = 'clay_ms' # 12 1e-3 1e-4
    # # train_opt.mat_name = 'cloth_ms'
    # # train_opt.mat_name = 'egyptian_statue_ms'
    # # train_opt.mat_name = 'face_ms'
    # # train_opt.mat_name = 'fake_and_real_beers_ms'
    # train_opt.mat_name = 'feathers_ms'
    
    train_opt.scale_factor = 4 #4 #8
    train_opt.num_theta = 30
    train_opt.core_tensor_dim = 64
    train_opt.print_freq = 1
    train_opt.save_freq = 1 #100
    train_opt.batchsize = 1
    train_opt.which_epoch = train_opt.niter + train_opt.niter_decay
    # train_opt.which_epoch = 20000
    # train_opt.continue_train = True
    train_opt.attention_use = True
    train_opt.useSoftmax = 'No' # No
    train_opt.isCalSP = 'Yes'
    train_opt.concat = 'Yes'
    train_opt.display_port = 8097

    # trade-off parameters: could be better tuned
    # for auto-reconstruction
    train_opt.lambda_A = 0.1
    train_opt.lambda_B = 1e-1 # 1e-3 # 1e-2 # spectral manifold 
    train_opt.lambda_C = 1e-2 # 1e-4 # 1e-3 # spatial manifold
    train_opt.lambda_F = 100

    train_dataloader = get_dataloader(train_opt, isTrain=True)
    dataset_size = len(train_dataloader)
    print(f"Dataset size {dataset_size}")
    train_opt.gpu_ids = [0]
    print("gpu opt", train_opt.gpu_ids)
    train_model = create_model(train_opt, train_dataloader.hsi_channels,
                               train_dataloader.msi_channels,
                               train_dataloader.lrhsi_height,
                               train_dataloader.lrhsi_width,
                               train_dataloader.sp_matrix,
                               train_dataloader.sp_range)

    train_model.setup(train_opt)

    print("current model isTrain", train_model.isTrain)

    print(f"Using device {train_model.device}")
    visualizer = Visualizer(train_opt, train_dataloader.sp_matrix)

    total_steps = 0

    print("Entering epoch loop")
    print(f"Epoch count starting at {train_opt.epoch_count}")
    print(train_opt.niter + train_opt.niter_decay + 1)

    checkpoint = None
    # checkpoint = './checkpoints'

    if checkpoint is None:
      print("No checkpoint - starting training from scratch")
      log_dir = f'./checkpoints/{train_opt.name}_x{train_opt.scale_factor}/'
      print(log_dir)
      if not os.path.exists(log_dir):
            os.mkdir(log_dir)
      filename = log_dir+f"{train_opt.name}_x{train_opt.scale_factor}.pth"

    if checkpoint is not None:
      filename = checkpoint
      print(f'Using check_point: {checkpoint}')
      # cp = torch.load(checkpoint)
      # train_model.load_state_dict(cp['model'],strict=False)  
      # train_model.optimizers.load_state_dict(cp['optimizer']) 
      # train_model.schedulers.load_state_dict(cp['scheduler'])
      # hist_batch_loss = cp['hist_batch_loss']
      # hist_epoch_loss = cp['hist_epoch_loss']
      train_model.load_networks(2720)

    # torch.cuda.empty_cache()


    for epoch in tqdm(range(train_opt.epoch_count, train_opt.which_epoch+ 1)):
    
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        train_psnr_list = []

        for i, data in enumerate(train_dataloader):

            iter_start_time = time.time()
            total_steps += train_opt.batchsize
            epoch_iter += train_opt.batchsize

            visualizer.reset()

            print("Visualizer reset")

            train_model.set_input(data, isTrain=True)
            train_model.optimize_joint_parameters(epoch)

            # hues.info("[{}/{} in {}/{}]".format(i, dataset_size // train_opt.batchsize,
            #                                     epoch, train_opt.niter + train_opt.niter_decay))

            train_psnr = train_model.cal_psnr()
            train_psnr_list.append(train_psnr)

            if epoch % train_opt.print_freq == 0:
                losses = train_model.get_current_losses()
                t = (time.time() - iter_start_time) / train_opt.batchsize
                visualizer.print_current_losses(epoch, epoch_iter, losses, t)
                if train_opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, train_opt, losses)
                    visualizer.display_current_results(train_model.get_current_visuals(),
                                                       train_model.get_image_name(), epoch, True,
                                                       win_id=[1])

                    visualizer.plot_spectral_lines(train_model.get_current_visuals(), train_model.get_image_name(),
                                                   visual_corresponding_name=train_model.get_visual_corresponding_name(),
                                                   win_id=[2, 3])
                    visualizer.plot_psnr_sam(train_model.get_current_visuals(), train_model.get_image_name(),
                                             epoch, float(epoch_iter) / dataset_size,
                                             train_model.get_visual_corresponding_name())

                    visualizer.plot_lr(train_model.get_LR(), epoch)
                
            # if epoch % 100 == 0:
            #     rec_hhsi = train_model.get_current_visuals()[train_model.get_visual_corresponding_name()['real_hhsi']].data.cpu().float().numpy()[0]
            #     sio.savemat(os.path.join("./checkpoints/" + train_opt.name  + "/results/", ''.join(data['name']) + '_' + str(epoch) + '.mat'), {'out': rec_hhsi.transpose(1, 2, 0)})

            # if epoch % (100*train_opt.print_freq) == 0:
            #     train_model.save_networks(epoch)

        if epoch % train_opt.save_epoch_freq == 0 :
            # state = {
            #     'epoch': epoch,
            #     'model': train_model.state_dict(),
            #     'optimizer': train_model.optimizers.state_dict(),
            #     'scheduler': train_model.schedulers.state_dict()
            #     # 'hist_batch_loss': hist_batch_loss,
            #     # 'hist_epoch_loss': hist_epoch_loss
            #     }
            # print(f"Saving: {filename}")   
            # save_checkpoint(state, filename=filename)
            train_model.save_networks(epoch)


        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, train_opt.niter + train_opt.niter_decay, time.time() - epoch_start_time))

        train_model.update_learning_rate()

    rec_hhsi = train_model.get_current_visuals()[train_model.get_visual_corresponding_name()['real_hhsi']].data.cpu().float().numpy()[0]
    # sio.savemat(os.path.join("./checkpoints/" + train_opt.name  + "/results/", ''.join(data['name']) + '_' + str(epoch) + '.mat'), {'out': rec_hhsi.transpose(1, 2, 0)})
    # sio.savemat(os.path.join("./Results/" + train_opt.name  + "/", ''.join(data['name']) + '.mat'), {'out': rec_hhsi.transpose(1, 2, 0)})

    print('full time %d sec' % (time.time() - start_time))
