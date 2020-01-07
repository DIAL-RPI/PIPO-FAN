# The official code of "Multi-organ Segmentation over Partially Labeled Datasets with Multi-scale Feature Abstraction"

## Introduction
In this paper, we propose a novel network architecture for unified multi-scale feature abstraction, which incorporates multi-scale features in a hierarchical fashion at various depths for image segmentation. 
The 2D network shows very competitive performance compared with other 3D networks in liver CT image segmentation with a single step. 
We further develop a unified segmentation strategy to train the three separate datasets together and do multi-organ segmentation with these partial datasets. It gives the segmentation network more robustness and accuracy. We'll test the method further in future work.
For more details, please refer to our [paper](https://arxiv.org/pdf/2001.00208.pdf).

## Instrction
- set up your environment by anaconda, (**python3.7, torch 1.3.0**)
- conda install -c simpleitk simpleitk
- conda install -c conda-forge nibabel
- pip install torchvision=0.4

## Pyramid Input Pyramid Output Feature Abstraction Network (PIPO-FAN)
To train PIPO-FAN on single dataset, modify the folder_training, folder_validation, checking_dir and log_dir path according to your local environment.

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
In this section the network can be trained over multiple datasets. Similar to last section, but the combination of multiple datasets are used for training.

## Citation
```
@article{fang2020multi,
  title={Multi-organ Segmentation over Partially Labeled Datasets with Multi-scale Feature Abstraction},
  author={Fang, Xi and Yan, Pingkun},
  journal={arXiv preprint arXiv:2001.00208},
  year={2020}
}
```
