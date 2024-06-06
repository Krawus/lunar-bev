"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

from fire import Fire

import src


if __name__ == '__main__':
    Fire({
        'cumsum_check': src.explore.cumsum_check,
        'eval_model_iou': src.explore.eval_model_iou,
        'load_data' : src.load_data.load_data,
        'train_lunar': src.train_lunar.train,
        'viz_lunar_preds': src.explore.viz_lunar_preds,
    })