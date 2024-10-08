"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from time import time
from tensorboardX import SummaryWriter
import numpy as np
import os

from .models import compile_model
from .lunarsim_data import compile_trainval_data, compile_test_data, compile_data_tts
from .tools import SimpleLoss, get_batch_iou, get_val_info, denormalize_img

# 1px -> 0.16m
# 1m -> 6.24px
def train(
            dataroot='/LunarSim',
            nepochs=100000,
            gpuid=0,

            H=768, W=1366,
            resize_lim=(0.193, 0.225),
            final_dim=(128, 352),
            bot_pct_lim=(0.0, 0.22),
            rot_lim=(-5.4, 5.4),
            rand_flip=True,
            ncams=6,
            max_grad_norm=5.0,
            pos_weight=2.13,
            logdir='./runs',
            from_checkpoint=False,
            checkpoint_path='./runs/checkpoints',
            use_tts=False,

            xbound=[-50.0, 50.0, 0.5],
            ybound=[-50.0, 50.0, 0.5],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[1.0, 42.0, 1.0],

            bsz=4,
            nworkers=8,
            lr=1e-3,
            # lr=1e-4,
            weight_decay=1e-7,
            # weight_decay=1e-8
            ):
    
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                    'Ncams': ncams,
                }
    
    if not use_tts:
        trainloader, valloader = compile_trainval_data(dataroot, data_aug_conf=data_aug_conf,
                                                    grid_conf=grid_conf, bsz=bsz, nworkers=nworkers)
    else:
        trainloader, valloader = compile_data_tts(dataroot, data_aug_conf=data_aug_conf,
                                                grid_conf=grid_conf, bsz=bsz, nworkers=nworkers)


    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    model = compile_model(grid_conf, data_aug_conf, outC=1)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = SimpleLoss(pos_weight).cuda(gpuid)

    writer = SummaryWriter(logdir=logdir)
    val_step = len(trainloader)

    counter = 0
    start_epoch = 0


    if from_checkpoint == True:
        if os.path.isfile(checkpoint_path):
            model_state = load_checkpoint(checkpoint_path, model, opt)
            print('Checkpoint found at iteration: ', model_state['iteration'])
        else:
            exit('Checkpoint not found')

        counter = model_state['iteration']
        start_epoch = model_state['epoch']

        print('Continuing training from checkpoint at iteration: ', counter)
        print('Starting epoch: ', start_epoch)


    model.train()
    
    for epoch in range(start_epoch, nepochs):
        np.random.seed()
        print('Epoch: ', epoch)
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(trainloader):
            t0 = time()
            opt.zero_grad()
            preds = model(imgs.to(device),
                    rots.to(device),
                    trans.to(device),
                    intrins.to(device),
                    post_rots.to(device),
                    post_trans.to(device),
                    )
            
            binimgs = binimgs.to(device)
            loss = loss_fn(preds, binimgs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            counter += 1
            t1 = time()

            if counter % 10 == 0:
                print("iteration: ", counter, "loss: ", loss.item())
                writer.add_scalar('train/loss', loss, counter)

            if counter % 50 == 0:
                _, _, iou = get_batch_iou(preds, binimgs)
                writer.add_scalar('train/iou', iou, counter)
                writer.add_scalar('train/epoch', epoch, counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)

            if counter % val_step == 0:
                val_info = get_val_info(model, valloader, loss_fn, device)
                print('VAL', val_info)
                writer.add_scalar('val/loss', val_info['loss'], counter)
                writer.add_scalar('val/iou', val_info['iou'], counter)

            if counter % (val_step*10) == 0:
                model.eval()
                mname = os.path.join(logdir, "model{}.pt".format(counter))
                print('saving', mname)
                torch.save(model.state_dict(), mname)
                print('saving checkpoint...')
                save_checkpoint(model, opt, epoch, logdir + '/checkpoints', counter)
                model.train()



# Saving the checkpoint
def save_checkpoint(model, optimizer, epoch, logdir, counter):
    state = {
        'iteration' : counter,
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    filepath = os.path.join(logdir, f'checkpoint_{counter}_epoch_{epoch}.pth')
    torch.save(state, filepath)



def load_checkpoint(filepath, model, optimizer):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint

