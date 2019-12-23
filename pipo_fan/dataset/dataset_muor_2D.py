#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 14:10:33 2017

@author: yanrpi
"""

# %%
import glob
import numpy as np
import nibabel as nib
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from os import path
# from scipy.misc import imsave
from scipy import ndimage
# from scipy.misc import imsave
# %%

class LiverCTDataset(Dataset):
    """Liver CT image dataset."""

    def __init__(self, root_dir, transform=None, verbose=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        if not path.isdir(root_dir):
            raise ValueError("\"{}\" is not a valid directory path!".format(root_dir))
            
        self.root_dir = root_dir
        self.transform = transform
        self.verbose = verbose
        
        res = glob.glob(path.join(root_dir, 'volume-*.nii'))
        #print(res)
        self.num_images = len(res)
        
        self.ct_filenames = res

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        img_name = self.ct_filenames[idx]
        seg_name = img_name.replace('volume', 'segmentation')
        image = nib.load(img_name)
        segmentation = nib.load(seg_name)
        # image = nib.as_closest_canonical(image)
        # segmentation = nib.as_closest_canonical(segmentation)
        
        if self.verbose:
            print('{} -> {}'.format(idx, img_name))
            print('Image shape: {}'.format(image.shape))
            print('Segmentation shape: {}'.format(segmentation.shape))
            
        sample = {'image': image, 'label': segmentation}
        #sample = {'image': img_name, 'segmentation': seg_name}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
# %%


class RandomCrop(object):
    """Crop randomly the image in a sample.
    For segmentation training, only crop sections with non-zero label

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size, view):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size
        self.view = view

    def __call__(self, sample):
        image, segmentation = sample['image'], sample['label']

        h, w, d = image.shape
        new_h, new_w, new_d = self.output_size
        view = self.view
        new_d_half = new_d >> 1

        # Find slices containing segmentation object
        seg_data = segmentation.get_data()
        img_data = image.get_data()
        if view == 'axial':
            img_data = img_data
            seg_data = seg_data
        elif view == 'coronal':
            img_data = img_data.transpose((2, 0, 1))
            seg_data = seg_data.transpose((2, 0, 1))
        else:
            img_data = img_data.transpose((2, 1, 0))
            seg_data = seg_data.transpose((2, 1, 0))
        summed = np.sum(seg_data.sum(axis=0), axis=0)
        non0_list = np.asarray([i for i in range(summed.size)])
        non0_list = non0_list[summed > 10]
        
        seg_start = max(np.min(non0_list) - new_d_half, 0)
        seg_end = min(np.max(non0_list) + new_d_half, d)
        if new_h == h:
            top = 0
            left = 0
        else:
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
        #ant = np.random.randint(0, d - new_d)
        ant = np.random.randint(seg_start, seg_end - new_d)

        
        img_data = img_data[top: top + new_h, 
                            left: left + new_w,
                            ant: ant + new_d]
        img_data = img_data.astype(np.float32)
        
        ant_seg = ant + new_d_half
        seg_data = seg_data[top: top + new_h, 
                            left: left + new_w,
                            ant_seg: ant_seg + 1]
        # seg_data = seg_data[top: top + new_h, 
        #                     left: left + new_w,
        #                     ant: ant + new_d]
        seg_data = seg_data.astype(np.float32)
        # Merge labels
        # seg_data[seg_data > 1] = 1

        # flip up side down to correct
        # image = np.flip(img_data, axis=1).copy()
        # label = np.flip(seg_data, axis=1).copy()
        
        return {'image': img_data, 'label': seg_data}


class RandomHorizontalFlip(object):
    """Randomly flip the image in the horizontal direction.
    """

    def __call__(self, sample):
        
        if random.uniform(0,1) < 0.5:
            return sample
        
        # else return flipped sample
        image, label = sample['image'], sample['label']

        image = np.flip(image, axis=0).copy()
        label = np.flip(label, axis=0).copy()

        return {'image': image, 'label': label}

class RandomVerticalFlip(object):
    """Randomly flip the image in the horizontal direction.
    """

    def __call__(self, sample):
        
        if random.uniform(0,1) < 0.5:
            return sample
        
        # else return flipped sample
        image, label = sample['image'], sample['label']

        image = np.flip(image, axis=1).copy()
        label = np.flip(label, axis=1).copy()

        return {'image': image, 'label': label}
    
# def pixel_mask(image, p):
#     p_map = np.random.random(size = image.shape)
#     mask = p_map <= p
#     return mask
    
# def boundary_mask(label, p1, p2):
#     d_map_in = ndimage.distance_transform_edt(label)
#     label_r = 1 - label
#     d_map_out = ndimage.distance_transform_edt(label_r)
#     d_map = d_map_in + d_map_out
#     d_map[d_map<=3] = 1
#     d_map[d_map>3] = 0
#     # d_map = d_map<=5
#     # print('d_map:',d_map.sum())
#     p_map = d_map
#     p_map[p_map == 1] = p1
#     p_map[p_map == 0] = p2
#     # print('p_map:',(p_map==p1).sum())
#     r_map = np.random.random(size = label.shape)
#     mask = r_map <= p_map
#     mask = 1*mask
#     return mask

# def bkg_mask(label, p1, p2):
#     p_map = label.copy()
#     p_map[p_map>=1] = 1
#     p_map[p_map<1] = 0
#     # print('P_map.sum0',(p_map==0).sum())
#     # print('P_map.sum1',(p_map==1).sum())
#     p_map[p_map == 0] = p2
#     # print('p_mapsum1',p_map.sum())
#     p_map[p_map == 1] = p1
#     # print('p_map:',(p_map==p1).sum())
#     r_map = np.random.random(size = label.shape)
    
#     mask = r_map <= p_map
#     mask = 1*mask
#     # print('mask.sum:',mask.sum())
#     return mask

# def bdy2blk(bdy, nrows, ncols, p1, p2):
#     # print(bdy.shape)
#     bdy1 = np.squeeze(bdy,-1)
#     # 224 x 224
#     h, w = bdy1.shape
#     # print(h,nrows,h/nrows)
#     # 16 x 16 x 14 x 14
#     bdy1 = bdy1.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols)
#     bdy1 = bdy1.reshape(nrows, ncols, int(h/nrows), int(w/nrows))
#     # print('bdy1.shape:',bdy1.shape)
#     for i in range(bdy1.shape[0]):
#         for j in range(bdy1.shape[1]):
#             if bdy1[i][j].sum() >= 1:
#                 if np.random.random_sample() <= p1:
#                     bdy1[i][j] = np.ones(bdy1[i][j].shape)
#                 else:
#                     bdy1[i][j] = np.zeros(bdy1[i][j],shape)
#             else:
#                 if np.random.random_sample() <= p2:
#                     bdy1[i][j] = np.ones(bdy1[i][j].shape)
#                 else:
#                     bdy1[i][j] = np.zeros(bdy1[i][j].shape)
#     return bdy1

# def blk_mask(label, p1, p2):
#     d_map_in = ndimage.distance_transform_edt(label)
#     label_r = 1 - label
#     d_map_out = ndimage.distance_transform_edt(label_r)
#     d_map = d_map_in + d_map_out
#     d_map[d_map<=5] = 1
#     d_map[d_map>5] = 0
#     p_map = d_map
#     # print('p_map_shape:', p_map.shape)
#     mask = bdy2blk(p_map,16,16, p1, p2)
#     # p_map size 16 x 16 x 14 x 14
#     # p_map[p_map == 1] = p1
#     # p_map[p_map == 0] = p2
#     # r_map = np.random.random(size = label.shape)
#     # mask = r_map <= p_map 
#     # 16x16 --> 224 x 224
#     # print('mask_shape1', mask.shape)
#     mask = np.hstack(mask)
#     mask = np.hstack(mask)
#     # print('mask_shape', mask.shape)
#     mask = np.expand_dims(mask, -1)
#     return mask

# class BdyblkOut(object):
#     def __init__(self, probability1, probability2):
        
#         self.pa = probability1
#         self.pb = probability2

#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#         p1 = self.pa + (1 - self.pa) * np.random.random_sample()
#         p2 = self.pb + (1 - self.pb) * np.random.random_sample()
#         # mask = boundary_mask(label, p1, p2)
#         mask = bdyblk_mask(label, p1, p2)
#         # print('mask:',mask.shape)
#         image = image * mask
#         label = label * mask
        
        
#         return {'image': image, 'label': label, 'mask': mask}


# class BoundaryOut(object):
#     def __init__(self, probability1, probability2):
        
#         self.pa = probability1
#         self.pb = probability2

#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#         p1 = self.pa + (1 - self.pa) * np.random.random_sample()
#         p2 = self.pb + (1 - self.pb) * np.random.random_sample()
#         # p1 = self.pa
#         # p2 = self.pb
#         mask = boundary_mask(label, p1, p2)
#         # mask = bdyblk_mask(label, p1, p2)
#         # print('mask_:',mask.sum())
#         # noise = np.random.normal(0,0.33,image.shape)
#         # noise[noise>1] = 1
#         # noise[noise<-1] = -1
#         # noise = noise*(1-mask)

#         image = image * mask
#         # image = image
#         # image = image + noise

#         # label = label * mask
        
        
#         return {'image': image, 'label': label, 'mask': mask}

# class BkgOut(object):
#     def __init__(self, probability1, probability2):
        
#         self.pa = probability1
#         self.pb = probability2

#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#         p1 = self.pa + (1 - self.pa) * np.random.random_sample()
#         p2 = self.pb + (1 - self.pb) * np.random.random_sample()
#         mask = bkg_mask(label, p1, p2)
#         # print('mask:',mask.shape)
#         image = image * mask
#         label = label * mask
        
        
#         return {'image': image, 'label': label, 'mask': mask}

# class MaskOut(object):
#     def __init__(self, probability):
        
#         self.pb = probability

#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#         p = self.pb + (1 - self.pb) * np.random.random_sample()
#         mask = pixel_mask(image, p)
#         # print('mask:',mask.shape)
#         image = image * mask
#         label = label * mask
        
        
#         return {'image': image, 'label': label, 'mask': mask}

class Clip(object):
    """Clip the intensity values.

    Args:
        Lower and upper bounds.
    """

    def __init__(self, lower_bound, upper_bound):
        '''
        '''
        # Make sure upper bound is larger than the lower bound
        self.LB = min(lower_bound, upper_bound)
        self.UB = max(lower_bound, upper_bound)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image[image>self.UB] = self.UB
        image[image<self.LB] = self.LB
        
        return {'image': image, 'label': label}
    
    
class Normalize(object):
    """Normalize the input data to 0 mean 1 std per channel"""
    
    def __init__(self, lower_bound, upper_bound):
        self.LB = min(lower_bound, upper_bound)
        self.UB = max(lower_bound, upper_bound)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        #img_mean = np.mean(image, axis=(0,1))
        #img_std = np.std(image, axis=(0,1))
        
        #nc = image.shape[2]
        #for c in range(nc):
        #    image[:,:,c] = (image[:,:,c] - img_mean[c]) / img_std[c]
        mid_point = (self.LB + self.UB) / 2.0
        image -= mid_point
        half_range = (self.UB - self.LB) / 2.0
        image /= (half_range + 0.000001)
    
        return {'image': image, 'label': label}
    
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # image, label, mask = sample['image'], sample['label'], sample['mask']

        # swap color axis because
        # numpy image: W x H x C
        # torch image: C X H X W
        image = image.transpose((2, 1, 0))
        #print(image.shape, type(image), image.dtype)
        label = label.transpose((2, 1, 0))
        # mask = mask.transpose(2, 1, 0)
        #print(label.shape, type(label), label.dtype)

        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}
        # return {'image': torch.from_numpy(image),
        #         'label': torch.from_numpy(label),
        #         'mask': torch.from_numpy(mask)}
    

def get_composed_transform(hw, slices, view):
    composed = transforms.Compose([RandomCrop((hw, hw, slices),view),
                                   Clip(-200, 200),
                                   Normalize(-200, 200),
                                   RandomHorizontalFlip(),
                                   RandomVerticalFlip(),
                                #    MaskOut(0.5),
                                #    BoundaryOut(0.5, 1),
                                #    BdyblkOut(1, 0.5),
                                #    BkgOut(1,0.5),
                                   ToTensor()])

    return composed


# %% Tester

if __name__ == '__main__':
    img_folder = '/zion/fangx2/BTCV/training_256'
    #img_folder = '/Users/yan/Documents/data/LITS_training'
    log_dir = path.expanduser('/zion/fangx2/mu_or/train/logs/')
    
    composed = get_composed_transform(224, 3, 'axial')
    
    dataset = LiverCTDataset(img_folder, 
                             transform=composed,
                             verbose = True)
    '''
    for i in range(5):
        sample = dataset[i]
        img = sample['image']
        print(i, img.size(), type(img))
        label = sample['label']
        print(i, label.size(), type(label))
    '''
    
    # num_workers = 4 to use more processes
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True,
                            num_workers=0)
    
    #for i_batch, sample_batched in enumerate(dataloader):
    batch_it = iter(dataloader)
    sample_batched = next(batch_it)
    image_batch = sample_batched['image']
    label_batch = sample_batched['label']

    print('Batch size: {}, image size: {}, label size: {}'.format(len(image_batch), 
          image_batch.size(2),
          label_batch.size(2)))
    
    img_data = image_batch[0,0,:,:].numpy()
    v_min = img_data.min()
    v_max = img_data.max()
    print('Img -> max: {}, min: {}'.format(v_max, v_min))
    img_data = (img_data - v_min) / (v_max - v_min) * 255
    img_data = img_data.astype(np.uint8)
    
    label_data = label_batch[0,0,:,:].numpy()
    v_min = label_data.min()
    v_max = label_data.max()
    print('Label -> max: {}, min: {}'.format(v_max, v_min))
    label_data *= 255
    lable_data = label_data.astype(np.uint8)

    
    # Save images
    imsave(path.join(log_dir, 'image_sample.png'), img_data, format='png')
    imsave(path.join(log_dir, 'label_sample.png'), label_data, format='png')