#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 5 16:00:33 2017

@author: yan
"""

# %% train the network
import argparse
import datetime
import math
import numpy as np
import os
from os import path
import shutil
import time

import torch
from torch import cuda
from torch import optim
#from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from collections import OrderedDict
from torch.nn import init

import dataset.dataset_liverCT_2D as dl1
import dataset.dataset_muor_2D as dl2
# from model.denseu_net import DenseUNet
# from model.unet import UNet
from model.concave_dps_w import ResUNet
# from model.concave_res_w3 import ResUNet
# from model.resu_net import ResUNet
# from model.concave_dcc import ResUNet
#from model.concave_sh import ResUNet
# from scipy.misc import imsave

# %%

parser = argparse.ArgumentParser(description='PyTorch ResUNet Training')
parser.add_argument('--epochs', default=4000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batchsize', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 8)')
parser.add_argument('--blocksize', default=224, type=int,
                    metavar='N', help='H/W of each image block (default: 224)')
parser.add_argument('-s', '--slices', default=3, type=int,
                    metavar='N', help='number of slices (default: 3)')
parser.add_argument('--lr', '--learning-rate', default=0.0002, type=float,
                    metavar='LR', help='initial learning rate (default: 0.002)')
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='N', help='momentum for optimizer (default: 0.9)')
parser.add_argument('--view', default='axial', type=str,
                    metavar='View', help='view for segmentation (default: axial)')
parser.add_argument('--cv_n', default='1', type=str,
                    help='Cross validation Dataset num')

# %%

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

# %%

# def dice_similarity(output, target):
#     """Computes the Dice similarity"""
#     #batch_size = target.size(0)

#     smooth = 0.00001
    
#     # max returns values and positions
#     seg_channel = output.max(dim=1)[1]
#     seg_channel = seg_channel.float()
    
#     target = target.float()
    
#     #print('Shapes: {}, {}'.format(seg_channel.shape, target.shape))

#     intersection = (seg_channel * target).sum(dim=2).sum(dim=1)
#     union = (seg_channel + target).sum(dim=2).sum(dim=1)

#     dice = 2. * intersection / (union + smooth)
#     #print(intersection, union, dice)
#     return torch.mean(dice)



def dice_similarity(output, target):
    """Computes the Dice similarity"""
    #batch_size = target.size(0)

    smooth = 0.00001
    
    # max returns values and positions
    # output = output>0.5
    output = output.float()    
    target = target.float()

    seg_channel = output.view(output.size(0), -1)
    target_channel = target.view(target.size(0), -1)
    
    #print('Shapes: {}, {}'.format(seg_channel.shape, target.shape))

    intersection = (seg_channel * target_channel).sum()
    union = (seg_channel + target_channel).sum()

    dice = (2. * intersection) / (union + smooth)
    #print(intersection, union, dice)
    return torch.mean(dice)
        
def dice_similarity_u(output, target):
    """Computes the Dice similarity"""
    #batch_size = target.size(0)
    total_dice = 0
    output = output.clone()
    target = target.clone()
    # print('target:',target.sum())
    for i in range(1, output.shape[1]):
        target_i = torch.zeros(target.shape)
        target_i = target_i.cuda().clone()
        target_i[target == i] = 1
        output_i = output[:, i:i+1].clone()

        dice_i = dice_similarity(output_i, target_i)
        # print('dice_: ',i,dice_i.data)
        # print('target_i: ',target_i.sum())
        # print('output_i: ',output_i.sum())
        total_dice += dice_i
    total_dice = total_dice / (output.shape[1] - 1)
    #print(intersection, union, dice)
    return total_dice
        
def visualize_train(d,name):
    name = name
    da = d.cpu().data.numpy()
    db = np.transpose(da[0], (1,2,0))
    # print('db.shape',db.shape)
    if db.shape[2] == 3:
        imsave(path.join('/home/fangx2/mu_or/train_u', name+'.png'), db, format='png')
    else:
        imsave(path.join('/home/fangx2/mu_or/train_u', name+'.png'), db[:,:,0], format='png')

def visualize_train1(d,name):
    name = name
    da = d.cpu().data.numpy()
    db = da[0,:,:]
    imsave(path.join('/home/fangx2/mu_or/train_u', name+'.png'), db, format='png')     

def visualize_val(d,name):
    name = name
    da = d.cpu().data.numpy()
    db = np.transpose(da[0], (1,2,0))
    # print('db.shape',db.shape)
    if db.shape[2] == 3:
        imsave(path.join('/home/fangx2/mu_or/val_u', name+'.png'), db, format='png')
    else:
        imsave(path.join('/home/fangx2/mu_or/val_u', name+'.png'), db[:,:,0], format='png')

def visualize_val1(d,name):
    name = name
    da = d.cpu().data.numpy()
    db = da[0,:,:]
    imsave(path.join('/home/fangx2/mu_or/val_u', name+'.png'), db, format='png')     
    
# %%

def train(train_loader, data_type, model, criterion, optimizer, epoch, verbose=True):
    """Function for training"""
    batch_time = AverageMeter()
    #data_time = AverageMeter()
    losses = AverageMeter()
    dice = AverageMeter()

    # switch to train mode
    model.train()

    end_time = time.time()
    for i, sample_batched in enumerate(train_loader):
        # measure data loading time
        #data_time.update(time.time() - end_time)

        image_batch = sample_batched['image']
        # label should be only the middle slice
        label_batch = sample_batched['label'][:,0,:,:]
        # mask = sample_batched['mask'][:,0:1,:,:]
        # print('mask shape:', mask.shape)
        #print('label batch size: {}'.format(label_batch.shape))
        #image_batch = image_batch.cuda()
        #label_batch = label_batch.cuda(async=True)
        input_var = Variable(image_batch).float()
        input_var = input_var.cuda()
        target_var = Variable(label_batch).long()
        target_var = target_var.cuda()
        # mask_var = Variable(mask).float()
        # mask_var = mask_var.cuda()

        # compute output
        output = model(input_var)
        output = torch.clamp(output, min=1e-10, max=1)
        
        if data_type == '1':
            output_p2 = output[:,1:2,:,:].clone()
            output_p1 = output[:,0:1,:,:].clone() + output[:,2:3,:,:].clone() + output[:,3:4,:,:].clone()
            output_p = torch.cat((output_p1, output_p2), 1)
        if data_type == '2':
            output_p2 = output[:,2:3,:,:].clone()
            output_p1 = output[:,0:1,:,:].clone() + output[:,1:2,:,:].clone() + output[:,3:4,:,:].clone()
            output_p = torch.cat((output_p1, output_p2), 1)
        if data_type == '3':
            output_p2 = output[:,3:4,:,:].clone()
            output_p1 = output[:,0:1,:,:].clone() + output[:,1:2,:,:].clone() + output[:,2:3,:,:].clone()
            output_p = torch.cat((output_p1, output_p2), 1)
        if data_type == '4':
            output_p = output.clone()
        # print('output p:',output_p.sum())
        # output = output * mask_var
        # print('Output size: {}, type: {}'.format(output.size(), type(output)))
        # print('Target size: {}, type: {}'.format(target_var.size(), type(target_var)))
        loss = criterion(output_p, target_var)

        # if epoch % 5 == 0:
        #     visualize_train(output_p[:,1:4,:,:], str(epoch) + 'output')
        #     visualize_train1(target_var[:,:,:], str(epoch) + 'target')

        # measure accuracy and record loss
        losses.update(loss.data, image_batch.size(0))
        ds = dice_similarity_u(output_p, target_var)
        #print(ds.data)
        dice.update(ds.data, image_batch.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        current_time = time.time()
        batch_time.update(current_time - end_time)
        end_time = current_time

        if ((i+1) % 10 == 0) and verbose:
            print('Train ep {0} [batch {1}/{2}]: '
                  #'Time {batch_time.val:.1f}s, '
                  'Loss avg: {loss.avg:.4f}, '
                  'Dice avg: {dice.avg:.4f}'.format(
                      epoch+1, i+1, len(train_loader),
                      #batch_time=batch_time,
                      loss=losses,
                      dice=dice))

    print('Training -> loss: {loss.avg:.4f}, '
          'Dice {dice.avg:.3f}'.format(
              loss=losses, dice=dice))

    #return {'train_loss': loss.avg, 'train_acc': dice.avg}
    return losses.avg, dice.avg


# %%

def validate(loader, data_type, model, criterion, epoch, verbose=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    dice = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, sample_batched in enumerate(loader):

        image_batch = sample_batched['image']
        # label should be only the middle slice
        label_batch = sample_batched['label'][:,0,:,:]
        # mask = sample_batched['mask'][:,0:1,:,:]
        
        input_var = Variable(image_batch, volatile=True).float()
        input_var = input_var.cuda()
        target_var = Variable(label_batch, volatile=True).long()
        target_var = target_var.cuda()
        # mask_var = Variable(mask).float()
        # mask_var = mask_var.cuda()


        # compute output
        output = model(input_var)
        # output = output * mask_var
        if data_type == '1':
            output_p = output[:,0:2,:,:].clone()
            output_p[:,0,:,:] = output[:,0,:,:].clone() + output[:,2,:,:].clone() + output[:,3,:,:].clone()
        if data_type == '2':
            output_p = output[:,1:3,:,:].clone()
            output_p[:,0,:,:] = output[:,0,:,:].clone() + output[:,1,:,:].clone() + output[:,3,:,:].clone()
        if data_type == '3':
            output_p = output[:,2:4,:,:].clone()
            output_p[:,0,:,:] = output[:,0,:,:].clone() + output[:,1,:,:].clone() + output[:,2,:,:].clone()
        if data_type == '4':
            output_p = output.clone()

        # if epoch % 5 == 0:
        #     visualize_val(output_p[:,1:4,:,:], str(epoch) + 'output')
        #     visualize_val1(target_var[:,:,:], str(epoch) + 'target')

        loss = criterion(output_p, target_var)

        #torch.save(input_var, '/home/yanp2/tmp/resu-net/logs/input_{}.pth'.format(i))
        #torch.save(target_var, '/home/yanp2/tmp/resu-net/logs/target_{}.pth'.format(i))
        #torch.save(output, '/home/yanp2/tmp/resu-net/logs/output_{}.pth'.format(i))

        # measure accuracy and record loss
        losses.update(loss.data, image_batch.size(0))
        ds = dice_similarity_u(output_p, target_var)
        dice.update(ds.data, image_batch.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if ((i+1) % 10 == 0) and verbose:
            print('Validation ep {0} [batch {1}/{2}]: '
                  #'Time {batch_time.val:.1f}s, '
                  'Loss avg: {loss.avg:.4f}, '
                  'Dice avg: {dice.avg:.4f}'.format(
                      epoch+1, i+1, len(loader),
                      #batch_time=batch_time,
                      loss=losses,
                      dice=dice))

    print('Validation ep {} -> loss: {loss.avg:.4f}, '
          'Dice {dice.avg:.3f}'.format(
              epoch+1, loss=losses, dice=dice))

    return losses.avg, dice.avg


#def adjust_learning_rate(optimizer, epoch):
def adjust_learning_rate(optimizer, gamma=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= gamma


        
# %%
def save_checkpoint(state, is_best, log_folder, view='axial',
                    filename='checkpoint.pth.tar'):
    """Save checkpoints
    """
    filename = path.join(log_folder, filename)
    torch.save(state, filename)
    if is_best:
        filename_best = path.join(log_folder, 'resu_best_{}.pth.tar'.format(view))
        shutil.copyfile(filename, filename_best)

        
# def compute_length(inputs, edge_op):
#     """Compute the length of segmentation boundary"""
    
#     # Get segmentation
#     seg_channel = inputs.max(dim=1)[1]
#     seg_channel = seg_channel.unsqueeze(1)
#     seg_channel = seg_channel.float()
#     #print(seg_channel.shape)

#     g2 = F.conv2d(seg_channel, edge_op, padding=1)
#     gx = g2 ** 2
#     gx = torch.sum(torch.squeeze(gx), dim=0)
#     # Adding small number to increase the numerical stability
#     #gx = torch.sqrt(gx + 1e-16)
#     gm = torch.mean(gx.view(-1))
    
#     return gm
    

# class HybridLoss2d(nn.Module):
#     def __init__(self, edge_op, weight=None, size_average=True):
#         super(HybridLoss2d, self).__init__()
#         self.nll_loss = nn.NLLLoss2d(weight, size_average)
#         self.op = edge_op

#     def forward(self, inputs, targets):
#         #return self.nll_loss(F.log_softmax(inputs, dim=1), targets)
        
#         ce = self.nll_loss(F.log_softmax(inputs, dim=1), targets)
        
#         # dice
#         dice = dice_similarity(inputs, targets)
        
#         # boundary length
#         length = compute_length(inputs, self.op)
        
#         return ce - 0.1 * dice + length

    
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(torch.log(inputs), targets)
    
# class FocalLoss2d(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(FocalLoss2d, self).__init__()
#         self.nll_loss = nn.NLLLoss2d(weight, size_average)

#     def forward(self, inputs, targets):
#         focal_frequency =  F.nll_loss(F.softmax(inputs, dim=1), targets, reduction = 'none')
#         # print('shape1:',focal_frequency.shape)
#         focal_frequency += 1.0
#         focal_frequency = torch.pow(focal_frequency, 2)
#         focal_frequency = focal_frequency.repeat(2, 1, 1, 1)
#         focal_frequency = focal_frequency.transpose(1,0)
#         # print('shape:',focal_frequency.shape)
#         return self.nll_loss(focal_frequency * F.log_softmax(inputs, dim=1), targets)
# %%

if __name__ == "__main__":
    
    global args
    args = parser.parse_args()
    cv = args.cv_n
    use_cuda = cuda.is_available()
    
    checkpoing_dir = path.expanduser('/home/fangx2/mu_or/tmp/sf_134')
    if not path.isdir(checkpoing_dir):
        os.makedirs(checkpoing_dir)
        
    log_dir = path.expanduser('/home/fangx2/mu_or/tmp/sf_134')
    if not path.isdir(log_dir):
        os.makedirs(log_dir)

    """
    training
    """
    num_classes = 4
    num_in_channels = args.slices
    # model = DenseUNet(num_channels = num_in_channels, num_classes = num_classes)
    model = ResUNet(num_in_channels, num_classes)
    # model = UNet(num_in_channels, num_classes)

    resunet_checkpoint = torch.load('/home/fangx2/mu_or/tmp/sf_pr0_1216_dps/resunet_checkpoint_final.pth.tar')
    resunet_dict = resunet_checkpoint['state_dict']

    model.resnet.load_state_dict(resunet_dict)

    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum)

    folder_training_1 = '/home/fangx2/data/LIver_submit1/data3/training_a/'
    folder_validation_1 = '/home/fangx2/data/LIver_submit1/data3/training_a/'

    folder_training_2 = '/home/fangx2/kits19/training_256_ras_a/'
    folder_validation_2 = '/home/fangx2/kits19/training_256_ras_a/'

    folder_training_3 = '/home/fangx2/data/code/data/spleen/training_a/'
    folder_validation_3 = '/home/fangx2/data/code/data/spleen/training_a/'

    folder_training_4 = '/home/fangx2/BTCV/training_256/'
    folder_validation_4 = '/home/fangx2/BTCV/validation_256/'


    # folder_training = r'/home/fangx2/data/LIver_submit1/dataset_256'
    # folder_validation = r'/home/fangx2/data/LIver_submit1/dataset_256'
    
    # Set L2 penalty using weight_decay
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # optimizer = optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    # Initialize Sobel edge detection filter
    sobel_x = np.asarray([1.0, 0, -1.0, 2.0, 0, -2.0, 1.0, 0, -1.0], dtype=np.float32)
    sobel_x /= 4.0
    sobel_x = np.reshape(sobel_x, (1, 1, 3, 3))

    sobel_y = np.asarray([1.0, 2.0, 1.0, 0, 0, 0, -1.0, -2.0, -1.0], dtype=np.float32)
    sobel_y /= 4.0
    sobel_y = np.reshape(sobel_y, (1, 1, 3, 3))

    sobel = np.concatenate((sobel_x, sobel_y), axis=0)
    sobel = Variable(torch.from_numpy(sobel), requires_grad=False)

    if use_cuda:
        sobel = sobel.cuda()

    # weights = torch.Tensor([0.2, 1.2])

    #Cross entropy Loss
    criterion = CrossEntropyLoss2d()
    # criterion = FocalLoss2d(weights)
    #criterion = HybridLoss2d(sobel, weights)
    
    if use_cuda:
        print('\n***** Training ResU-Net with GPU *****\n')
        model.cuda()
        criterion.cuda()

    blocksize = args.blocksize
    view = args.view
    if view == 'axial' or view == 'sagittal' or view == 'coronal':
        composed = dl1.get_composed_transform(blocksize, num_in_channels, view)
        composed4 = dl2.get_composed_transform(blocksize, num_in_channels, view)
    else:

        print('The given view of <{}> is not supported!'.format(view))


    batchsize = args.batchsize

#Dataset 1,2,3
    dataset_train1 = dl1.LiverCTDataset(folder_training_1,
                                      transform=composed)
    train_loader1 = dl1.DataLoader(dataset_train1,
                                 batch_size=args.batchsize,
                                 shuffle=True,
                                 num_workers=4,
                                 drop_last=False
                                )

    dataset_validation1 = dl1.LiverCTDataset(folder_validation_1,
                                           transform=composed)
    val_loader1 = dl1.DataLoader(dataset_validation1,
                               batch_size=args.batchsize,
                               shuffle=False,
                               num_workers=2,
                               drop_last=False
                              )

    # dataset_train2 = dl1.LiverCTDataset(folder_training_2,
    #                                   transform=composed)
    # train_loader2 = dl1.DataLoader(dataset_train2,
    #                              batch_size=args.batchsize,
    #                              shuffle=True,
    #                              num_workers=4,
    #                              drop_last=False
    #                             )

    # dataset_validation2 = dl1.LiverCTDataset(folder_validation_2,
    #                                        transform=composed)
    # val_loader2 = dl1.DataLoader(dataset_validation2,
    #                            batch_size=args.batchsize,
    #                            shuffle=False,
    #                            num_workers=2,
    #                            drop_last=False
    #                           )

    dataset_train3 = dl1.LiverCTDataset(folder_training_3,
                                      transform=composed)
    train_loader3 = dl1.DataLoader(dataset_train3,
                                 batch_size=args.batchsize,
                                 shuffle=True,
                                 num_workers=4,
                                 drop_last=False
                                )

    dataset_validation3 = dl1.LiverCTDataset(folder_validation_3,
                                           transform=composed)
    val_loader3 = dl1.DataLoader(dataset_validation3,
                               batch_size=args.batchsize,
                               shuffle=False,
                               num_workers=2,
                               drop_last=False
                              )
#Dataset4
    dataset_train4 = dl2.LiverCTDataset(folder_training_4,
                                      transform=composed4)
    train_loader4 = dl2.DataLoader(dataset_train4,
                                 batch_size=args.batchsize,
                                 shuffle=True,
                                 num_workers=4,
                                 drop_last=False
                                )

    dataset_validation4 = dl2.LiverCTDataset(folder_validation_4,
                                           transform=composed4)
    val_loader4 = dl2.DataLoader(dataset_validation4,
                               batch_size=args.batchsize,
                               shuffle=False,
                               num_workers=2,
                               drop_last=False
                              )
    



    best_dice = -1.0
    #for epoch in range(args.start_epoch, args.epochs):
    num_epochs = args.epochs
    
    train_history = []
    val_history = []
    
    for epoch in range(num_epochs):
        print('Training epoch {} of {}...'.format(epoch + 1, num_epochs))
        # start timing
        t_start = time.time()
    
        # train for one epoch
        if epoch % 3 == 0:
            train_loss = train(train_loader1, '1', model, criterion, 
                            optimizer, epoch, verbose=True)
        elif epoch % 3 == 1:
        #     train_loss = train(train_loader2, '2', model, criterion, 
        #                     optimizer, epoch, verbose=True)
        # # elif epoch % 4 == 2:
        # else:
            train_loss = train(train_loader3, '3', model, criterion, 
                            optimizer, epoch, verbose=True)
        else:
            train_loss = train(train_loader4, '4', model, criterion, 
                            optimizer, epoch, verbose=True)

        # train_loss = train(train_loader4, '4', model, criterion, 
        #                     optimizer, epoch, verbose=True)
                            
        train_history.append(train_loss)
        
        # Gradually reducing learning rate
        if epoch % 40 == 0:
            adjust_learning_rate(optimizer, gamma=0.99)

        # evaluate on validation set
        val_loss = validate(val_loader4, '4', model, criterion, epoch, verbose=True)
        val_history.append(val_loss)

        dice = val_loss[1]
        # remember best prec@1 and save checkpoint
        is_best = dice > best_dice
        best_dice = max(dice, best_dice)

        if is_best:
            fn_checkpoint = 'resu_checkpoint_ep{:04d}.pth.tar'.format(epoch + 1)
            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model.state_dict(),
                             'best_dice': best_dice,
                             'optimizer' : optimizer.state_dict(),},
                            is_best,
                            checkpoing_dir,
                            view,
                            filename=fn_checkpoint)
        if epoch == num_epochs - 1:
            filename = path.join(checkpoing_dir, 'resunet_checkpoint_final.pth.tar')
            torch.save({'epoch': epoch + 1,
                             'state_dict': model.state_dict(),
                             'best_dice': best_dice,
                             'optimizer' : optimizer.state_dict(),},filename)
        elapsed_time = time.time() - t_start
        print('Epoch {} completed in {:.2f}s\n'.format(epoch+1, elapsed_time))
        
    # save the training history
    time_now = datetime.datetime.now()
    time_str = time_now.strftime('%y%m%d-%H%M%S')

    fn_train_history = path.join(log_dir, 'train_hist_{}.npy'.format(time_str))
    fn_val_history = path.join(log_dir, 'val_hist_{}.npy'.format(time_str))
    
    np.save(fn_train_history, np.asarray(train_history))
    np.save(fn_val_history, np.asarray(val_history))
    
    time_disp_str = time_now.strftime('%H:%M:%S on %Y-%m-%d')
    print('Training completed at {}'.format(time_disp_str))
    print('Training history saved into:\n<{}>'.format(fn_train_history))
    print('<{}>'.format(fn_val_history))