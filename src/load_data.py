
import torch
from time import time
from tensorboardX import SummaryWriter
import numpy as np
import os

from .models import compile_model
from .lunarsim_data import compile_trainval_data, compile_data_tts
from .tools import SimpleLoss, get_batch_iou, get_val_info



def load_data(
            dataroot='/Lunarsim',
            nepochs=10000,
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

            xbound=[-16.0, 16.0, 0.16],
            ybound=[-16.0, 16.0, 0.16],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[0.3, 21.3, 0.5],

            bsz=3,
            nworkers=8,
            lr=1e-3,
            weight_decay=1e-7,
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
    # trainloader, valloader = compile_trainval_data(dataroot, data_aug_conf=data_aug_conf,
    #                                       grid_conf=grid_conf, bsz=bsz, nworkers=nworkers)

    print("COMPILING USING TTS")
    trainloader, valloader = compile_data_tts(dataroot, data_aug_conf=data_aug_conf,
                                            grid_conf=grid_conf, bsz=bsz, nworkers=nworkers)
    
    


    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')
    
    model = compile_model(grid_conf, data_aug_conf, outC=1)
    model.to(device)

    model.train()
    

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = SimpleLoss(pos_weight).cuda(gpuid)

    counter = 0
    for epoch in range(nepochs):
        np.random.seed()
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


