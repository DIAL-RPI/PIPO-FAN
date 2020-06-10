# Multi-organ Segmentation over Partially Labeled Datasets with Multi-scale Feature Abstraction

## Introduction
In this paper, we propose a novel network architecture for unified multi-scale feature abstraction, which incorporates multi-scale features in a hierarchical fashion at various depths for image segmentation. 
The 2D network shows very competitive performance compared with other 3D networks in liver CT image segmentation with a single step. 
We further develop a unified segmentation strategy to train the three separate datasets together and do multi-organ segmentation with these partial datasets. It gives the segmentation network more robustness and accuracy. We'll test the method further in future work.
For more details, please refer to our [IEEE TMI paper](https://doi.org/10.1109/TMI.2020.3001036) or the pre-print version available on [arXiv](https://arxiv.org/pdf/2001.00208.pdf).

## Instruction
- set up your environment by anaconda, (**python3.7, torch 1.3.0**)
- conda install -c simpleitk simpleitk
- conda install -c conda-forge nibabel
- pip install torchvision=0.4

## Preprocessing
Use resample.py to resample the size of each slice into 256*256. Use command line:
```
resample -p1 '/home/fangx2/data/preCT/' -p2 '/home/fangx2/data/prect_256/' -s1 256 -s2 256
```
## Pyramid Input Pyramid Output Feature Abstraction Network (PIPO-FAN)
To train PIPO-FAN on single dataset, modify the folder_training, folder_validation, checking_dir and log_dir path according to your local environment. To train the PIPO, use train_concave0.py. To train the PIPO-FAN, use train_sf_partial.

### Training
```
CUDA_VISIBLE_DEVICES=0 python train_concave0.py
```
```
CUDA_VISIBLE_DEVICES=0 python train_sf_partial.py
```
### Validation
```
CUDA_VISIBLE_DEVICES=0 python segment_sf.py --view 'axial' --slices 3 $img_dir $val_dir --evaluating True --label_filename $label_dir --network _path $network_path
```

## Multi-organ segmentation over multiple datasets
In this section the network can be trained over multiple datasets. Similar to last section, but the combination of multiple datasets are used for training. BTCV, LiTS, KiTS and spleen datasets are used in our experiments.

## Citation
```
@article{fang2020multi,
  title={Multi-organ Segmentation over Partially Labeled Datasets with Multi-scale Feature Abstraction},
  author={Fang, Xi and Yan, Pingkun},
  journal={IEEE Transactions on Medical Imaging},
  year={2020},
  volume={},
  number={},
  pages={1-1},
}
```
Pre-print version is available at [arXiv](https://arxiv.org/abs/2001.00208)
