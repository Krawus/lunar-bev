"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as mpatches
import numpy as np

# from .data import compile_data
from .tools import (ego_to_cam, get_only_in_img_mask, denormalize_img,
                    SimpleLoss, get_val_info, add_ego, gen_dx_bx)
from .models import compile_model
import cv2
from .lunarsim_data import compile_trainval_data, compile_test_data


def cumsum_check(version,
                dataroot='/data/nuscenes',
                gpuid=1,

                H=768, W=1366,
                resize_lim=(0.193, 0.225),
                final_dim=(128, 352),
                bot_pct_lim=(0.0, 0.22),
                rot_lim=(-5.4, 5.4),
                rand_flip=True,

              
                xbound=[-34.0, 34.0, 0.34],
                ybound=[-34.0, 34.0, 0.34],
                zbound=[-10.0, 10.0, 20.0],
                dbound=[1.0, 42.0, 1.0],

                bsz=4,
                nworkers=8,
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
                    'Ncams': 6,
                }
    trainloader, valloader = compile_trainval_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')
    loader = trainloader

    model = compile_model(grid_conf, data_aug_conf, outC=1)
    model.to(device)

    model.eval()
    for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(loader):

        model.use_quickcumsum = False
        model.zero_grad()
        out = model(imgs.to(device),
                rots.to(device),
                trans.to(device),
                intrins.to(device),
                post_rots.to(device),
                post_trans.to(device),
                )
        out.mean().backward()
        print('autograd:    ', out.mean().detach().item(), model.camencode.depthnet.weight.grad.mean().item())

        model.use_quickcumsum = True
        model.zero_grad()
        out = model(imgs.to(device),
                rots.to(device),
                trans.to(device),
                intrins.to(device),
                post_rots.to(device),
                post_trans.to(device),
                )
        out.mean().backward()
        print('quick cumsum:', out.mean().detach().item(), model.camencode.depthnet.weight.grad.mean().item())
        print()


def eval_model_iou(version,
                modelf,
                dataroot='/data/nuscenes',
                gpuid=0,

                H=768, W=1366,
                resize_lim=(0.193, 0.225),
                final_dim=(128, 352),
                bot_pct_lim=(0.0, 0.22),
                rot_lim=(-5.4, 5.4),
                rand_flip=True,
              
                xbound=[-34.0, 34.0, 0.34],
                ybound=[-34.0, 34.0, 0.34],
                zbound=[-10.0, 10.0, 20.0],
                dbound=[1.0, 42.0, 1.0],

                bsz=4,
                nworkers=8,
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
                    'Ncams': 6,
                }
    trainloader, valloader = compile_trainval_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers)

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    model = compile_model(grid_conf, data_aug_conf, outC=1)
    print('loading', modelf)
    model.load_state_dict(torch.load(modelf))
    model.to(device)

    loss_fn = SimpleLoss(1.0).cuda(gpuid)

    model.eval()
    val_info = get_val_info(model, valloader, loss_fn, device)
    print(val_info)


def viz_lunar_preds(version,
                    modelf,
                    dataroot='/home/krawus/nuscenes',
                    map_folder='/data/nuscenes/mini',
                    gpuid=0,
                    viz_train=False,

                    H=768, W=1366,
                    resize_lim=(0.193, 0.225),
                    final_dim=(128, 352),
                    bot_pct_lim=(0.0, 0.22),
                    rot_lim=(-5.4, 5.4),
                    rand_flip=True,

                    xbound=[-34.0, 34.0, 0.34],
                    ybound=[-34.0, 34.0, 0.34],
                    zbound=[-10.0, 10.0, 20.0],
                    dbound=[1.0, 42.0, 1.0],

                    bsz=3,
                    nworkers=8,
                    ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': cams,
                    'Ncams': 6,
                }
    # testloader = compile_test_data(dataroot, data_aug_conf=data_aug_conf,
    #                                       grid_conf=grid_conf, bsz=bsz, nworkers=nworkers)
    
    trainloader, valloader = compile_trainval_data(dataroot, data_aug_conf=data_aug_conf,
                                                   grid_conf=grid_conf, bsz=bsz, nworkers=nworkers)
    # loader = trainloader if viz_train else 
    # loader = testloader
    loader = valloader
    # nusc_maps = get_nusc_maps(map_folder)

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    model = compile_model(grid_conf, data_aug_conf, outC=1)
    print('loading', modelf)
    model.load_state_dict(torch.load(modelf))
    model.to(device)

    dx, bx, _ = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
    dx, bx = dx[:2].numpy(), bx[:2].numpy()


    val = 0.01
    fH, fW = final_dim
    fig = plt.figure(figsize=(3*fW*val, (1.5*fW + 2*fH)*val))
    gs = mpl.gridspec.GridSpec(3, 6, height_ratios=(1.5*fW, fH, fH))
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    model.eval()
    counter = 0
    with torch.no_grad():
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(loader):
            out = model(imgs.to(device),
                    rots.to(device),
                    trans.to(device),
                    intrins.to(device),
                    post_rots.to(device),
                    post_trans.to(device),
                    )
            out = out.sigmoid().cpu()

            

            for si in range(imgs.shape[0]):
                plt.clf()
                for imgi, img in enumerate(imgs[si]):
                    ax = plt.subplot(gs[1 + imgi // 3, (imgi % 3) * 2: (imgi % 3) * 2 + 2])
                    showimg = denormalize_img(img)
                    showimgcv = cv2.cvtColor(np.array(showimg), cv2.COLOR_RGB2BGR)
                    # cv2.imshow(f'img{cams[imgi]}', showimgcv)
                    # # cv2.imshow('showimg', )
                    # cv2.imshow('binimg vis', binimgs[si].squeeze(0).numpy())
                    # cv2.imshow('out', out[si].squeeze(0).numpy())
                    # cv2.waitKey(10000)
                    # flip the bottom images
                    if imgi > 2:
                        showimg = showimg.transpose(Image.FLIP_LEFT_RIGHT)
                    plt.imshow(showimg)
                    plt.axis('off')
                    plt.annotate(cams[imgi].replace('_', ' '), (0.01, 0.92), xycoords='axes fraction', color='red')

                # display network output
                ax = plt.subplot(gs[0, :3])
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                # border
                # plt.setp(ax.spines.values(), color='b', linewidth=2)
                plt.legend(handles=[
                    mpatches.Patch(color=(0.0, 0.0, 1.0, 1.0), label='Output Rock Segmentation'),
                    mpatches.Patch(color='#76b900', label='Ego Vehicle'),
                ], loc=(0.01, 0.86))

                
                # plt.imshow(np.rot90(out[si].squeeze(0), 2), vmin=0, vmax=1, cmap='Blues')
                plt.imshow(out[si].squeeze(0), vmin=0, vmax=1, cmap='Blues')
                add_ego(bx, dx)


                # display network ground truth output
                ax = plt.subplot(gs[0, 3:])
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                # border
                # plt.setp(ax.spines.values(), color='b', linewidth=2)
                plt.legend(handles=[
                    mpatches.Patch(color=(0.0, 0.0, 1.0, 1.0), label='Ground Truth Rock Segmentation'),
                    mpatches.Patch(color='#76b900', label='Ego Vehicle'),
                ], loc=(0.01, 0.86))
                # plt.imshow(np.rot90(binimgs[si].squeeze(0), 2), vmin=0, vmax=1, cmap='Blues')
                plt.imshow(binimgs[si].squeeze(0), vmin=0, vmax=1, cmap='Blues')
                add_ego(bx, dx)




                imname = f'eval{batchi:06}_{si:03}.jpg'
                print('saving', imname)
                plt.savefig('visualization_output/'+imname)
                counter += 1

