"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
import os
import numpy as np
from PIL import Image
import cv2
from pyquaternion import Quaternion
from glob import glob
from sklearn.model_selection import train_test_split
import json

from .tools import get_lidar_data, img_transform, normalize_img, gen_dx_bx, denormalize_img


class LunarData(torch.utils.data.Dataset):
    def __init__(self, dataroot, is_train, data_aug_conf, grid_conf, train_test_val):
        
        self.is_train = is_train
        self.dataroot = dataroot
        self.data_aug_conf = data_aug_conf
        self.cams = data_aug_conf['cams']

        self.cam_img_paths =  self.get_imgs_paths_by_cam(train_test_val)
        self.binimg_paths = self.get_gt_imgs_paths(train_test_val)
        self.indexes = self.select_indexes(train_test_val)
        self.cams_intrins = self.load_cam_intrinsics()
        self.cams_extrins = self.load_cam_extrinsics()

        self.grid_conf = grid_conf

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()



    def get_imgs_paths_by_cam(self, train_test_val):
        img_paths = {}
        for cam in self.cams:
                paths = sorted(glob(self.dataroot + train_test_val + '/solo*/sequence.0/step*.' + cam + '.png'))
                img_paths[cam] = paths
                if len(paths) == 0:
                    raise FileNotFoundError(f"{train_test_val} images not found for {cam}")
        return img_paths

    def get_gt_imgs_paths(self, train_test_val):
        paths = sorted(glob(self.dataroot + train_test_val + '/solo*/sequence.0/step*.semantic segmentation.png'))
        if len(paths) == 0:
            raise FileNotFoundError(f"{train_test_val} binary images not found")
        return paths

    def select_indexes(self, train_test_val):
        all_indexes = list(range(len(self.binimg_paths)))
        # train_indexes, val_indexes = train_test_split(all_indexes, test_size=0.2, random_state=42)

        # if train_test_val == 'train':
        #     return train_indexes
        # elif train_test_val == 'val':
        #     return val_indexes
        # elif train_test_val == 'test':
        #     return all_indexes
        # else:
        #     raise ValueError("train_test_val must be 'train', 'val', or 'test'")

        return all_indexes
        

    def load_cam_intrinsics(self):
        try:
            with open('LunarSim/cam_intrinsics.json', 'r') as f:
                intrins_data = json.load(f)
            return intrins_data
        except FileNotFoundError:
            raise FileNotFoundError("cam_intrinsics.json file not found")
        
    
    def load_cam_extrinsics(self):
        try:
            with open('LunarSim/cam_extrinsics.json', 'r') as f:
                extrins_data = json.load(f)
            return extrins_data
        except FileNotFoundError:
            raise FileNotFoundError("cam_extrinsics.json file not found")

    
    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH/H, fW/W)
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

 
    def get_imgs_data(self, idx, cams):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        for cam in cams:
            imgname = self.cam_img_paths[cam][idx]
            img = Image.open(imgname).convert("RGB")

            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            # sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            intrin = torch.Tensor(self.cams_intrins[cam])
            rot = torch.Tensor(Quaternion(self.cams_extrins[cam]['rotation']).rotation_matrix)
            tran = torch.Tensor(self.cams_extrins[cam]['translation'])


            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                     resize=resize,
                                                     resize_dims=resize_dims,
                                                     crop=crop,
                                                     flip=flip,
                                                     rotate=rotate,
                                                     )
            
            # # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2
            
            imgs.append(normalize_img(img))

            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))


    
    def get_gt_img(self, idx):
        imgname = self.binimg_paths[idx]
        img = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)
        img = self.cut_square_to_shortest_dim(img)
        img = cv2.resize(img, (self.nx[0], self.nx[1]))
        img[img > 0] = 1.0
        img = img.astype(float)

        return torch.Tensor(img).unsqueeze(0)

    def cut_square_to_shortest_dim(self, img):
        height, width = img.shape
        shorter_dim = min(height, width)
        start_x = (width - shorter_dim) // 2
        start_y = (height - shorter_dim) // 2
        end_x = start_x + shorter_dim
        end_y = start_y + shorter_dim
        cropped_image = img[start_y:end_y, start_x:end_x]
        
        return cropped_image

    def choose_cams(self):
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
        return cams

    def __str__(self):
        return f"""LunarData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.indexes)


class SegmentationData(LunarData):
    def __init__(self, *args, **kwargs):
        super(SegmentationData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        idx = self.indexes[index]
        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_imgs_data(idx, cams)
        binimg = self.get_gt_img(idx)

        return imgs, rots, trans, intrins, post_rots, post_trans, binimg


def worker_rnd_init(x):
    np.random.seed(13 + x)


def compile_trainval_data(dataroot, data_aug_conf, grid_conf, bsz, nworkers):

    traindata = SegmentationData(dataroot, is_train=True, data_aug_conf=data_aug_conf,
                         grid_conf=grid_conf, train_test_val='train')

    valdata = SegmentationData(dataroot, is_train=False, data_aug_conf=data_aug_conf,
                       grid_conf=grid_conf, train_test_val='val')
    

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                              shuffle=True,
                                              num_workers=nworkers,
                                              drop_last=True,
                                              worker_init_fn=worker_rnd_init)
    

    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)
    
    
 
    return trainloader, valloader



def compile_test_data(dataroot, data_aug_conf, grid_conf, bsz, nworkers):

    testdata = SegmentationData(dataroot, is_train=False, data_aug_conf=data_aug_conf,
                       grid_conf=grid_conf, train_test_val='test')
    
    
    
    testloader = torch.utils.data.DataLoader(testdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)

    return testloader
