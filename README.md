# LunarBev



This is the repository for the implementation of the BEV algorithm on the Moon.

The simulation environment was obtained thanks to [LunarSim](https://github.com/PUTvision/LunarSim) project.

The BEV implementation was modeled on the work of [Lift-Splat-Shoot](https://github.com/nv-tlabs/lift-splat-shoot/tree/master)

### Instalation
```
pip install -r requirements.txt
```

### Training
```
python main.py train_lunar --dataroot=PATH_TO_DATA --logdir=./runs --gpuid=0 
```

### Evaluate a model
```
python main.py eval_model_iou mini/trainval --modelf=PATH_TO_MODEL --dataroot=PATH_TO_DATA
```

### Visualize Predictions
```
python main.py viz_model_preds mini/trainval --modelf=PATH_TO_MODEL --dataroot=PATH_TO_DATA
```


## Final algorith result visualization
<img src="./imgs/lunarBev_viz.gif">

