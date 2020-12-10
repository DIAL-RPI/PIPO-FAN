#compute avg dice and global dice
import os
from os import path
import nibabel as nib
import numpy as np

gt_pth = '/zion/fangx2/data/test_segmentation_lits_submission/'
res_pth = '/zion/fangx2/data/LIver_submit1/val_pre_test_512/'

dice = []
weight = []
dataset = []
intersection = []
union = []

def compute_dice(la,lb):
    intersection = np.sum(la * lb)
    union = np.sum(la + lb)
    return intersection, union, 2 * intersection/(union+0.00001)

for j in range(0,121,1):
    
    j = str(j) + '.nii'
    # three dimension
    result = nib.load(path.join(res_pth,'test-segmentation-'+j))
    label = nib.load(path.join(gt_pth,'test-segmentation-'+j))
    
    result_data = result.get_data()
    result_data[result_data>1] = 1
    label_data = label.get_data()
    label_data[label_data>1] = 1
    
    intersection_i, union_i, dice_i = compute_dice(result_data,label_data)
    
    intersection.append(intersection_i)
    union.append(union_i)
    dice.append(dice_i)
    
    dataset.append(j)
    print('img:{}'.format(j),'2 * intersection:{}'.format(2 * intersection_i),'union:{}'.format(union_i))
    
dice_avg = float(sum(dice)) / len(dice)
dice_global = 2 * float(sum(intersection)) / float(sum(union))

# print(dataset)
# print(dice)
# print(weight)

print('the average dice: {} '.format(dice_avg))
print('the global dice: {} '.format(dice_global))


