#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 16:24:59 2017

@author: yan

Load pre-trained network to segment a new image

Code v0.01
"""

# %% Resnet blocks in U-net

import argparse
import datetime
import nibabel as nib
import numpy as np
import os
from os import path
from scipy import ndimage
import SimpleITK as sitk
import time

import torch
from torch import cuda
from torch import optim
from torch.autograd import Variable
import torch.nn as nn

# from unet_context import UNet_ctx
#from u_net import UNet
# from model.concave_dps import ResUNet
from model.concave_dps_w import ResUNet
# from model.concave_res2 import ResUNet
# from model.concave_res_w3 import ResUNet
#from fcoordresu_net import ResUNet
#from resu_ctx import ResUNet
# %%

parser = argparse.ArgumentParser(description='ResUNet CT segmentation')
parser.add_argument('input_filename', type=str, metavar='input_filename',
                    help='File of image to be segmented')
parser.add_argument('output_filename', type=str, metavar='output_filename',
                    help='File to save the segmentation result')
parser.add_argument('-s', '--slices', default=3, type=int,
                   help='number of slices (default: 5)')
parser.add_argument('--begin', default=0, type=int,
                    help='Beginning slice for segmentation')
parser.add_argument('--end', default=9999, type=int,
                    help='Ending slice for segmentation')
parser.add_argument('-c', '--cuda', default=True, type=bool, metavar='Use GPU CUDA',
                    help='Use GPU for computation')
parser.add_argument('-e', '--evaluating', default=False, type=bool,
                    metavar='evaluation after segmentation', help='Use GT label for evaluation after completing segmentation')
parser.add_argument('-l', '--label_filename', default=None, type=str,
                    metavar='label_filename',
                    help='File containing the ground truth segmentation label for evaluation')
parser.add_argument('--network_path', default='./', type=str,
                    metavar='path of network file',
                    help='File containing the pre-trained network')
parser.add_argument('--view', default='axial', type=str,
                    metavar='View', help='view for segmentation (default: axial)')

# %%

def load_image(image_filename, evaluating=False, label_filename=None):
    """
    """
    image = nib.load(image_filename)

    if evaluating and path.isfile(label_filename):
        label = nib.load(label_filename)
    else:
        label = None

    return {'image':image, 'label':label}

# %%

def load_network(fn_network, gpu=True):
    """ Load pre-trained network
    """
    if path.isfile(fn_network):
        print("=> loading checkpoint '{}'".format(fn_network))
        if gpu:
            checkpoint = torch.load(fn_network)
        else:
            checkpoint = torch.load(fn_network, map_location=lambda storage, loc: storage)

        # Currently only support binary segmentation
        # num_classes = 2
        #model = UNet(5,2)
        #model = UNet_ctx(3,5,2)
        model = ResUNet(3,4)
        model.load_state_dict(checkpoint['state_dict'])
        if gpu:
            model.cuda()
        else:
            model.cpu()

        # optimizer = optim.SGD(model.parameters(), lr=0.02)
        # if gpu:
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        # else:
        optimizer = None

        print("=> loaded checkpoint at epoch {}"
              .format(checkpoint['epoch']))

        return model, optimizer
    else:
        print("=> no checkpoint found at '{}'".format(fn_network))
        return None, None
# %%

def compute_dice(la, lb):
    intersection = np.sum(la * lb)
    union = np.sum(la + lb)
    return 2 * intersection / (union + 0.00001)

# %%

class SimpleITKAsNibabel(nib.Nifti1Image):
    """
    Minimal interface to use a SimpleITK image as if it were
    a nibabel object. Currently only supports the subset of the
    interface used by NiftyNet and is read only
    """

    def __init__(self, itk_image):
        #try:
        self._SimpleITKImage = itk_image
        #except RuntimeError as err:
        #    if 'Unable to determine ImageIO reader' in str(err):
        #        raise nibabel.filebasedimages.ImageFileError(str(err))
        #    else:
        #        raise
        # self._header = SimpleITKAsNibabelHeader(self._SimpleITKImage)
        affine = make_affine(self._SimpleITKImage)
        # super(SimpleITKAsNibabel, self).__init__(
        #     sitk.GetArrayFromImage(self._SimpleITKImage).transpose(), affine)
        nib.Nifti1Image.__init__(
            self,
            sitk.GetArrayFromImage(self._SimpleITKImage).transpose(), affine)


class SimpleITKAsNibabelHeader(nib.spatialimages.SpatialHeader):
    def __init__(self, image_reference):
        super(SimpleITKAsNibabelHeader, self).__init__(
            data_dtype=sitk.GetArrayViewFromImage(image_reference).dtype,
            shape=sitk.GetArrayViewFromImage(image_reference).shape,
            zooms=image_reference.GetSpacing())


def make_affine(simpleITKImage):
    # get affine transform in LPS
    c = [simpleITKImage.TransformContinuousIndexToPhysicalPoint(p)
         for p in ((1, 0, 0),
                   (0, 1, 0),
                   (0, 0, 1),
                   (0, 0, 0))]
    c = np.array(c)
    affine = np.concatenate([
        np.concatenate([c[0:3] - c[3:], c[3:]], axis=0),
        [[0.], [0.], [0.], [1.]]], axis=1)
    affine = np.transpose(affine)
    # convert to RAS to match nibabel
    affine = np.matmul(np.diag([-1., -1., 1., 1.]), affine)
    return affine

# %%

class Nifti_from_numpy(nib.Nifti1Image):
    """
    Minimal interface to use a SimpleITK image as if it were
    a nibabel object. Currently only supports the subset of the
    interface used by NiftyNet and is read only
    """

    def __init__(self, array, itk_image):
        #try:
        self._SimpleITKImage = itk_image
        #except RuntimeError as err:
        #    if 'Unable to determine ImageIO reader' in str(err):
        #        raise nibabel.filebasedimages.ImageFileError(str(err))
        #    else:
        #        raise
        # self._header = SimpleITKAsNibabelHeader(self._SimpleITKImage)
        affine = make_affine(self._SimpleITKImage)
        # super(SimpleITKAsNibabel, self).__init__(
        #     sitk.GetArrayFromImage(self._SimpleITKImage).transpose(), affine)
        nib.Nifti1Image.__init__(
            self, array.transpose(), affine)

def extract_volume(volume):
    volumes = []
    x_coord = []
    y_coord = []
    for x in range(0,volume.shape[1],112):
        for y in range(0,volume.shape[2],112):
            end_x = x + 224
            end_y = y + 224
            if end_x > volume.shape[1]:
                x = volume.shape[1] - 224
                end_x = volume.shape[1]
            if end_y > volume.shape[2]:
                y = volume.shape[2] - 224
                end_y = volume.shape[2]
            cur_img = volume[:, x:end_x, y:end_y]
            volumes.append(cur_img)
            x_coord.append(x)
            y_coord.append(y)
            if y == volume.shape[2] - 224:
                break
        if x == volume.shape[1] - 224:
            break
    return volumes, x_coord, y_coord

def construct_volume(volumes,x_coord, y_coord):
    x_len = max(x_coord) + 224
    y_len = max(y_coord) + 224
    seg_matrix = []
    mul_matrix = []
    for i in range(len(volumes)):
        output = torch.zeros([volumes[i].shape[0],volumes[i].shape[1],x_len,y_len],dtype=torch.float32)
        time_matrix = torch.zeros([volumes[i].shape[0],volumes[i].shape[1], x_len,y_len])
        x_start = x_coord[i]
        y_start = y_coord[i]
        x_end = x_start + 224
        y_end = y_start + 224
        output[:,:,x_start:x_end, y_start:y_end] = volumes[i]
        time_matrix[:,:, x_start:x_end, y_start:y_end] = torch.ones(volumes[i].shape)
        seg_matrix.append(output)
        mul_matrix.append(time_matrix)
    seg_matrix = torch.cat(seg_matrix,0)
    mul_matrix = torch.cat(mul_matrix,0)
    seg_matrix = torch.sum(seg_matrix, 0)
    mul_matrix = torch.sum(mul_matrix, 0)
    seg_final = torch.div(seg_matrix, mul_matrix)
    seg_final = seg_final.cuda()
    return seg_final


# %%

if __name__ == "__main__":

    args = parser.parse_args()
    evaluating = args.evaluating
    use_cuda = args.cuda
    slice_begin = args.begin
    slice_end = args.end
    view = args.view
    if not cuda.is_available():
        print('No available GPU can be used for computation!')
        use_cuda = False

    num_channels = args.slices
    # num_channels = 3

    #fn_network = path.expanduser('~/tmp/resu-net3D/checkpoints/resu3d_checkpoint_ep0578.pth.tar')
    #fn_network = path.join(args.network_path, 'resu_best.pth.tar')

    #load the trained best 2D model
    # fn_network = path.join(args.network_path,'resunet_checkpoint_final.pth.tar')
    fn_network = path.join(args.network_path,'resu_best_' + view + '.pth.tar')
    print('Loading network from <{}>'.format(fn_network))
    if not path.isfile(fn_network):
        raise Exception('Missing network <{}>! File Not Found!'.format(fn_network))

    model_axial, optimizer = load_network(fn_network, gpu=use_cuda)
    # Set model to evaluation mode
    model_axial.eval()

    #img_filename = path.expanduser(args.input_filename)
    #file in computer/home/data/ct_nih
    img_filename = args.input_filename
    print('Input image for segmentation:\t{}'.format(img_filename))

    dicom_input = False

    # Check if it is DICOM folder
    if path.isdir(img_filename):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames( img_filename )
        reader.SetFileNames(dicom_names)

        image = reader.Execute()
        dicom_input = True

        w, h, d = image.GetSize()

        img_data = sitk.GetArrayFromImage(image)
    else:
        volume = load_image(img_filename, evaluating, args.label_filename)

        image, label = volume['image'], volume['label']
        w, h, d = image.shape[:3]

        img_data = np.squeeze(image.get_data())

    print('Size of the input image: {}x{}x{}'.format(w, h, d))

    img_data = img_data.astype(np.float32)

    if view == 'axial':
        img_data = img_data
    elif view == 'coronal':
            img_data = img_data.transpose((2,0,1))
    else:
        img_data = img_data.transpose(2,1,0)

    img_data[img_data > 200] = 200.0
    img_data[img_data < -200] = -200.0
    img_data /= 200.0

    print('Segmenting image...')
    start_time = time.time()

    results = []

    num_half_channels = num_channels >> 1

    # Define the range of segmentation
    first = max(num_half_channels, slice_begin)
    last = min(d - num_half_channels - 1, slice_end)
    #last = min(d - num_channels + 1, slice_end)
    num_segmented_slices = last - first + 1
    print('Segmenting {} slices between [{}, {}]'.format(
        num_segmented_slices, first, last))

    for i in range(first):
        #results.append(np.zeros((1,1,w,h)))
        results.append(np.zeros((1,h,w)))

    #for depth in range(d - num_channels + 1):
    for depth in range(first - num_half_channels,
                       last - num_half_channels):

        if dicom_input:
            subvolume = img_data[depth:depth+num_channels,:,:]
        else:
            subvolume = img_data[:,:,depth:depth+num_channels]
            subvolume = subvolume.transpose((2, 1, 0))


        subvolumes, x_coor, y_coor = extract_volume(subvolume)

        outputs = []
        for volume in subvolumes:
            volume = volume[np.newaxis,:,:,:]
            volume = Variable(torch.from_numpy(volume), volatile=True).float()
            if use_cuda:
                volume = volume.cuda()

             #subs.append(subvolume)
            # output1, output2, output3, output4, output5 = model_axial(volume)
            output5 = model_axial(volume)
            # output_s = nn.Softmax2d()(output5)
            outputs.append(output5)

        output = construct_volume(outputs, x_coor, y_coor)

        output = output.max(dim=0)[1].cpu().data.numpy()
        output = output[np.newaxis,:,:]
        results.append(output)

        #results.append(output.cpu().data.numpy())

    print('It took {:.1f}s to segment {} slices'.format(
        time.time() - start_time, num_segmented_slices))



    #for i in range(num_half_channels):
    for i in range(d - last):
        #results.append(np.zeros((1,1,w,h)))
        results.append(np.zeros((1,h,w)))

    results = np.squeeze(np.asarray(results))
    #dsize = list(results.shape)
    c, h, w = results.shape
    #print('Segmentation result in CxHxW: {}x{}x{}'.format(c, h, w))
    if not dicom_input:
        if view == 'axial':
            results = np.transpose(results, (2, 1, 0))
        elif view == 'coronal':
            results = np.transpose(results,(1, 0, 2))
        else:
            results = results
    print('Segmentation result in HxWxC: {}x{}x{}'.format(h, w, c))

    # results[results > 0.49] = 1
    # results[results < 0.5] = 0
    results = results.astype(np.uint8)

    if evaluating:
        label_data = label.get_data()
        # remove tumor label
        label_data[label_data > 1] = 1
        dice = compute_dice(results, label_data)
        print('Dice score of ResU-Net: {:.3f}'.format(dice))

    # print('Starting morphological post-processing...')
    # #print('no postprocess...')
    # # perform morphological operation
    # #remove small noisy segmentation
    # results = ndimage.binary_opening(results, iterations=5)
    # #Generate smooth segmentation
    # results = ndimage.binary_dilation(results, iterations=3)
    # results = ndimage.binary_fill_holes(results)
    # results = ndimage.binary_erosion(results, iterations=3)

    # perform largest connected component analysis
    # labeled_array, num_features = ndimage.label(results)
    # size_features = np.zeros((num_features))
    # for i in range(num_features):
    #     size_features[i] = np.sum(labeled_array == i+1)
    # results = np.zeros_like(labeled_array)
    # results[labeled_array == np.argmax(size_features) + 1] = 1
    results_post = np.zeros_like(results)
    min_co = 0
    for i in range(1, 4):
        
        #liver
        if i ==1:
            results_i = np.zeros(results.shape)
            # results_i = results_i.cuda().clone()
            results_i[results == i] = 1
            labeled_array_i, num_features_i = ndimage.label(results_i)
            size_features_i = np.zeros((num_features_i))
            for j in range(num_features_i):
                size_features_i[j] = np.sum(labeled_array_i == j+1)
            results_i = np.zeros_like(labeled_array_i)
            results_i[labeled_array_i == np.argmax(size_features_i) + 1] = i
            results_i = results_i.astype(np.uint8)
            summed_1 = np.sum(results_i.sum(axis=0), axis=0)
            non0_list = np.asarray([i for i in range(summed_1.size)])
            non0_list = non0_list[summed_1 > 1]
            min_co = 0.8 * np.min(non0_list)
            min_co = int(min_co)
            print('min_co', min_co)
        #kidney
        if i == 2:
            results_i = np.zeros(results.shape)
            # results_i = results_i.cuda().clone()
            results_i[results == i] = 1
            results_i[:,:,:min_co] = 0
            labeled_array_i, num_features_i = ndimage.label(results_i)
            size_features_i = np.zeros((num_features_i))
            for j in range(num_features_i):
                size_features_i[j] = np.sum(labeled_array_i == j+1)
            results_i = np.zeros_like(labeled_array_i)
            # print('idx1:',np.argmax(size_features_i))
            results_i[labeled_array_i == np.argmax(size_features_i) + 1] = i
            results1_i = np.zeros_like(labeled_array_i)
            idx2 = np.argsort(-size_features_i)[1]
            # print('idx2:',idx2)
            results1_i[labeled_array_i == idx2 + 1] = i
            results_i = results_i + results1_i
            results_i = results_i.astype(np.uint8)
            
        #spleen
        else:
            results_i = np.zeros(results.shape)
            # results_i = results_i.cuda().clone()
            results_i[results == i] = 1
            results_i[:,:,:min_co] = 0
            labeled_array_i, num_features_i = ndimage.label(results_i)
            size_features_i = np.zeros((num_features_i))
            for j in range(num_features_i):
                size_features_i[j] = np.sum(labeled_array_i == j+1)
            results_i = np.zeros_like(labeled_array_i)
            results_i[labeled_array_i == np.argmax(size_features_i) + 1] = i
            results_i = results_i.astype(np.uint8)
        results_post += results_i

    results = results_post

    # results = results.astype(np.uint8)

    # Create the segmentation image for saving
    if dicom_input:
        new_image = Nifti_from_numpy(results, image)
    else:
        header = image.header
        header.set_data_dtype(np.uint8)

        # if nifty1
        if header['sizeof_hdr'] == 348:
            new_image = nib.Nifti1Image(results, image.affine, header=header)
        # if nifty2
        elif header['sizeof_hdr'] == 540:
            new_image = nib.Nifti2Image(results, image.affine, header=header)
        else:
            raise IOError('Input image header problem')

    #seg_dir = path.expanduser('~/tmp/resu-net/segmentation')
    #fn_seg = path.join(seg_dir, 'segmentation.nii')
    fn_seg = path.expanduser(args.output_filename)
    print('Writing segmentation result into <{}>...'.format(fn_seg))

    #mu.write_mhd_file(fn_seg, results, meta_dict=header)
    nib.save(new_image, fn_seg)
    print('Segmentation result has been saved.')

    # Compute Dice for evaluating
    if evaluating:
        dice = compute_dice(results, label_data)
        print('Final Dice score: {:.3f}'.format(dice))


