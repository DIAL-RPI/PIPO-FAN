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

# from lovasz_losses import lovasz_softmax

import dataset.dataset_liverCT_2D as dl
# import dataset.dataset_all as dl
#from u_net import UNet
# from model.concave_sh import ResUNet
# from model.MIMO_att import ResUNet
# from model.concave_res2 import ResUNet
from model.concave_dps import ResUNet
# from model.concave_dps_dc import ResUNet
# from model.concave_dps3 import ResUNet
#from resu_scalecov import ResUNet  
#from coordu_net import UNet

# %%

parser = argparse.ArgumentParser(description='PyTorch ResUNet Training')
parser.add_argument('--epochs', default=2000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batchsize', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 8)')
parser.add_argument('--blocksize', default=224, type=int,
                    metavar='N', help='H/W of each image block (default: 320)')
parser.add_argument('-s', '--slices', default=3, type=int,
                    metavar='N', help='number of slices (default: 1)')
parser.add_argument('-n', '--num_classes', default=2, type=int,
                    metavar='N', help='number of slices (default: 3)')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float,
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
def dice_similarity(output, target):
    """Computes the Dice similarity"""
    #batch_size = target.size(0)

    smooth = 0.00001
    
    # max returns values and positions
    seg_channel = output.max(dim=1)[1]
    seg_channel = seg_channel.float()
    
    target = target.float()
    
    #print('Shapes: {}, {}'.format(seg_channel.shape, target.shape))

    intersection = (seg_channel * target).sum(dim=2).sum(dim=1)
    union = (seg_channel + target).sum(dim=2).sum(dim=1)

    dice = 2. * intersection / (union + smooth)
    #print(intersection, union, dice)
    return torch.mean(dice)

# def dice_similarity(output, target):
#     """Computes the Dice similarity"""
#     #batch_size = target.size(0)

#     smooth = 0.00001
    
#     # max returns values and positions
#     output = output>0.5
#     output = output.float()    
#     target = target.float()

#     seg_channel = output.view(output.size(0), -1)
#     target_channel = target.view(target.size(0), -1)
    
#     #print('Shapes: {}, {}'.format(seg_channel.shape, target.shape))

#     intersection = (seg_channel * target_channel).sum()
#     union = (seg_channel + target_channel).sum()

#     dice = (2. * intersection) / (union + smooth)
#     #print(intersection, union, dice)
#     return torch.mean(dice)
        

    
# %%

def train(train_loader, model, criterion, optimizer, epoch, verbose=True):
    """Function for training"""
    batch_time = AverageMeter()
    #data_time = AverageMeter()
    losses = AverageMeter()
    dice = AverageMeter()

    losses_1 = AverageMeter()
    dice_1 = AverageMeter()
    losses_2 = AverageMeter()
    dice_2 = AverageMeter()
    losses_3 = AverageMeter()
    dice_3 = AverageMeter()
    losses_4 = AverageMeter()
    dice_4 = AverageMeter()
    losses_5 = AverageMeter()
    dice_5 = AverageMeter()
    # losses_6 = AverageMeter()
    # dice_6 = AverageMeter()


    # switch to train mode
    model.train()

    end_time = time.time()
    for i, sample_batched in enumerate(train_loader):
        # measure data loading time
        #data_time.update(time.time() - end_time)


        image_batch = sample_batched['image']
        # label should be only the middle slice
        label_batch1 = sample_batched['label'][:,0,:,:]
        
        input_var = Variable(image_batch, volatile=True).float()
        input_var = input_var.cuda()
        
        target_var = Variable(label_batch1, volatile=True).long().cuda()
        # target_var = nn.Upsample(size = [256,256], mode='bilinear', align_corners=True)(target_var)

        # compute output
        output1, output2, output3, output4, output5 = model(input_var)
        # print('output:',output1.shape,output2.shape,output3.shape)
        loss1 = criterion(output1, target_var)
        loss2 = criterion(output2, target_var)
        loss3 = criterion(output3, target_var)
        loss4 = criterion(output4, target_var)
        loss5 = criterion(output5, target_var)
        # loss6 = criterion(output6, target_var)

        # a = (output1 - output2 + 1) / 2
        # a_tar = (target_var1 - target_var2 + 1) / 2
        # loss4 = criterion(a, a_tar)

        # b = (output3 - output2 +1) / 2
        # b_tar = (target_var3 - target_var2 + 1) / 2
        # loss5 = criterion(b, b_tar)

        # loss = loss1 + loss2 + loss3 + 0.5 * loss4 + 0.5 * loss5
        loss = loss1 + loss2 + loss3 + loss4 + loss5

        # measure accuracy and record loss
        losses.update(loss.data, image_batch.size(0))
        losses_1.update(loss1.data, image_batch.size(0))
        losses_2.update(loss2.data, image_batch.size(0))
        losses_3.update(loss3.data, image_batch.size(0))
        losses_4.update(loss4.data, image_batch.size(0))
        losses_5.update(loss5.data, image_batch.size(0))
        # losses_6.update(loss6.data, image_batch.size(0))


        ds_1 = dice_similarity(output1, target_var)
        ds_2 = dice_similarity(output2, target_var)
        ds_3 = dice_similarity(output3, target_var)
        ds_4 = dice_similarity(output4, target_var)
        ds_5 = dice_similarity(output5, target_var)
        # ds_6 = dice_similarity(output6, target_var)
        #print(ds.data)
        dice_1.update(ds_1.data, image_batch.size(0))
        dice_2.update(ds_2.data, image_batch.size(0))
        dice_3.update(ds_3.data, image_batch.size(0))
        dice_4.update(ds_4.data, image_batch.size(0))
        dice_5.update(ds_5.data, image_batch.size(0))
        # dice_6.update(ds_6.data, image_batch.size(0))

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

    print('Training -> loss: {loss.avg:.4f}'.format(
              loss=losses))

    print('Training -> loss_1: {loss.avg:.4f}, '
          'Dice_1 {dice_1.avg:.3f}'.format(
              loss=losses_1, dice_1=dice_1))
    print('Training -> loss_2: {loss.avg:.4f}, '
          'Dice_2 {dice_2.avg:.3f}'.format(
              loss=losses_2, dice_2=dice_2))
    print('Training -> loss_3: {loss.avg:.4f}, '
          'Dice_3 {dice_3.avg:.3f}'.format(
              loss=losses_3, dice_3=dice_3))

    print('Training -> loss_4: {loss.avg:.4f}, '
          'Dice_4 {dice_4.avg:.3f}'.format(
              loss=losses_4, dice_4=dice_4))
    print('Training -> loss_5: {loss.avg:.4f}, '
          'Dice_5 {dice_5.avg:.3f}'.format(
              loss=losses_5, dice_5=dice_5))
    # print('Training -> loss_6: {loss.avg:.4f}, '
    #       'Dice_6 {dice_6.avg:.3f}'.format(
    #           loss=losses_5, dice_6=dice_6))

    #return {'train_loss': loss.avg, 'train_acc': dice.avg}
    return losses.avg, dice_5.avg


# %%

def validate(loader, model, criterion, epoch, verbose=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    dice = AverageMeter()

    losses_1 = AverageMeter()
    dice_1 = AverageMeter()
    losses_2 = AverageMeter()
    dice_2 = AverageMeter()
    losses_3 = AverageMeter()
    dice_3 = AverageMeter()
    losses_4 = AverageMeter()
    dice_4 = AverageMeter()
    losses_5 = AverageMeter()
    dice_5 = AverageMeter()
    # losses_6 = AverageMeter()
    # dice_6 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, sample_batched in enumerate(loader):

        image_batch = sample_batched['image']
        # label should be only the middle slice
        label_batch1 = sample_batched['label'][:,0,:,:]
        
        input_var = Variable(image_batch, volatile=True).float()
        input_var = input_var.cuda()
        
        target_var = Variable(label_batch1, volatile=True).long().cuda()
        # compute output
        output1, output2, output3, output4, output5 = model(input_var)
        loss1 = criterion(output1, target_var)
        loss2 = criterion(output2, target_var)
        loss3 = criterion(output3, target_var)
        loss4 = criterion(output4, target_var)
        loss5 = criterion(output5, target_var)
        # loss6 = criterion(output6, target_var)
        

        # a = (output1 - output2 + 1) / 2
        # a_tar = (target_var1 - target_var2 + 1) / 2
        # loss4 = criterion(a, a_tar)

        # b = (output3 - output2 +1) / 2
        # b_tar = (target_var3 - target_var2 + 1) / 2
        # loss5 = criterion(b, b_tar)

        # loss = loss1 + loss2 + loss3 + 0.5 * loss4 + 0.5 * loss5
        loss = loss1 + loss2 + loss3 + loss4 + loss5

        #torch.save(input_var, '/home/yanp2/tmp/resu-net/logs/input_{}.pth'.format(i))
        #torch.save(target_var, '/home/yanp2/tmp/resu-net/logs/target_{}.pth'.format(i))
        #torch.save(output, '/home/yanp2/tmp/resu-net/logs/output_{}.pth'.format(i))


        # measure accuracy and record loss
        # measure accuracy and record loss
        losses.update(loss.data, image_batch.size(0))
        losses_1.update(loss1.data, image_batch.size(0))
        losses_2.update(loss2.data, image_batch.size(0))
        losses_3.update(loss3.data, image_batch.size(0))
        losses_4.update(loss4.data, image_batch.size(0))
        losses_5.update(loss5.data, image_batch.size(0))
        # losses_6.update(loss6.data, image_batch.size(0))
    
        ds_1 = dice_similarity(output1, target_var)
        ds_2 = dice_similarity(output2, target_var)
        ds_3 = dice_similarity(output3, target_var)
        ds_4 = dice_similarity(output4, target_var)
        ds_5 = dice_similarity(output5, target_var)
        # ds_6 = dice_similarity(output6, target_var)
        
        dice_1.update(ds_1.data, image_batch.size(0))
        dice_2.update(ds_2.data, image_batch.size(0))
        dice_3.update(ds_3.data, image_batch.size(0))
        dice_4.update(ds_4.data, image_batch.size(0))
        dice_5.update(ds_5.data, image_batch.size(0))
        # dice_6.update(ds_6.data, image_batch.size(0))

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

    print('Validation -> loss_1: {loss.avg:.4f}, '
          'Dice_1 {dice_1.avg:.3f}'.format(
              loss=losses_1, dice_1=dice_1))
    print('Validation -> loss_2: {loss.avg:.4f}, '
          'Dice_2 {dice_2.avg:.3f}'.format(
              loss=losses_2, dice_2=dice_2))
    print('Validation -> loss_3: {loss.avg:.4f}, '
          'Dice_3 {dice_3.avg:.3f}'.format(
              loss=losses_3, dice_3=dice_3))
    print('Validation -> loss_4: {loss.avg:.4f}, '
          'Dice_4 {dice_4.avg:.3f}'.format(
              loss=losses_4, dice_4=dice_4))
    print('Validation -> loss_5: {loss.avg:.4f}, '
          'Dice_5 {dice_5.avg:.3f}'.format(
              loss=losses_5, dice_5=dice_5))
    # print('Validation -> loss_6: {loss.avg:.4f}, '
    #       'Dice_6 {dice_6.avg:.3f}'.format(
    #           loss=losses_6, dice_6=dice_6))



    return losses.avg, dice_5.avg


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

        
def compute_length(inputs, edge_op):
    """Compute the length of segmentation boundary"""
    
    # Get segmentation
    seg_channel = inputs.max(dim=1)[1]
    seg_channel = seg_channel.unsqueeze(1)
    seg_channel = seg_channel.float()
    #print(seg_channel.shape)

    g2 = F.conv2d(seg_channel, edge_op, padding=1)
    gx = g2 ** 2
    gx = torch.sum(torch.squeeze(gx), dim=0)
    # Adding small number to increase the numerical stability
    #gx = torch.sqrt(gx + 1e-16)
    gm = torch.mean(gx.view(-1))
    
    return gm
    

class HybridLoss2d(nn.Module):
    def __init__(self, edge_op, weight=None, size_average=True):
        super(HybridLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)
        self.op = edge_op

    def forward(self, inputs, targets):
        #return self.nll_loss(F.log_softmax(inputs, dim=1), targets)
        
        ce = self.nll_loss(F.log_softmax(inputs, dim=1), targets)
        
        # dice
        dice = dice_similarity(inputs, targets)
        
        # boundary length
        length = compute_length(inputs, self.op)
        
        return ce - 0.1 * dice + length

    
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)

class LovaszLoss2d(nn.Module):
    def forward(self, inputs, targets):
        return lovasz_softmax(F.softmax(inputs), targets)

class LoCeLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(LoCeLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return lovasz_softmax(F.softmax(inputs), targets) + self.nll_loss(F.log_softmax(inputs, dim=1), targets)
    
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    def forward(self, input, target):
        smooth = 0.00001
        input = input.float()
        target = target.float()
        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        
        return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))

class FocalLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        focal_frequency =  F.nll_loss(F.softmax(inputs, dim=1), targets, reduction = 'none')
        # print('shape1:',focal_frequency.shape)
        focal_frequency += 1.0
        focal_frequency = torch.pow(focal_frequency, 2)
        focal_frequency = focal_frequency.repeat(2, 1, 1, 1)
        focal_frequency = focal_frequency.transpose(1,0)
        # print('shape:',focal_frequency.shape)
        return self.nll_loss(focal_frequency * F.log_softmax(inputs, dim=1), targets)

# %%

if __name__ == "__main__":
    
    global args
    args = parser.parse_args()
    cv = args.cv_n
    view = args.view
    use_cuda = cuda.is_available()
    
    # checkpoing_dir = path.expanduser('/home/fangx2/data/LIver_submit1/data' + cv + '/tmp/spleen_dps_1105')
    # checkpoing_dir = path.expanduser('/home/fangx2/data/code/data/spleen/spleen_dps_1105')
    # checkpoing_dir = path.expanduser('/home/fangx2/data/LIver_submit1/data3/tmp/liver_ras_1106')
    # if not path.isdir(checkpoing_dir):
    #     os.makedirs(checkpoing_dir)
        
    # # log_dir = path.expanduser('/home/fangx2/data/LIver_submit1/data' + cv + '/tmp/spleen_dps_1105')
    # # log_dir = path.expanduser('/home/fangx2/data/code/data/spleen/spleen_dps_1105')
    # log_dir = path.expanduser('/home/fangx2/data/LIver_submit1/data3/tmp/liver_ras_1106')
    # if not path.isdir(log_dir):
    #     os.makedirs(log_dir)
        
    checkpoing_dir = path.expanduser('/home/fangx2/data/code/data/spleen/5_fold_cv/fold' + cv + '/tmp/concave')
    if not path.isdir(checkpoing_dir):
        os.makedirs(checkpoing_dir)
        
    log_dir = path.expanduser('/home/fangx2/data/code/data/spleen/5_fold_cv/fold' + cv + '/tmp/concave')
    if not path.isdir(log_dir):
        os.makedirs(log_dir)
    """
    training
    """
    num_classes = args.num_classes
    num_in_channels = args.slices
    #model = UNet(5, 2)
    model = ResUNet(num_in_channels,num_classes)

    # resunet_checkpoint = torch.load('/home/fangx2/data/LIver_submit1/data' + cv + '/tmp/concave_dps_pre/resu_best_axial.pth.tar')
    # resunet_dict = resunet_checkpoint['state_dict']

    # model.load_state_dict(resunet_dict)

    # folder_training = r'/home/fangx2/data/LIver_submit1/data3/training_ras'
    # folder_validation = r'/home/fangx2/data/LIver_submit1/data3/validation_ras'
    folder_training = '/home/fangx2/data/code/data/spleen/5_fold_cv/fold' + cv + '/training/'
    folder_validation = '/home/fangx2/data/code/data/spleen/5_fold_cv/fold' + cv + '/validation/'
    # folder_training = r'/home/fangx2/data/code/data/spleen/training'
    # folder_validation = r'/home/fangx2/data/code/data/spleen/validation'
    # folder_training = r'/home/fangx2/data/LIver_submit1/data' + cv + '/training/'
    # folder_validation = r'/home/fangx2/data/LIver_submit1/data' + cv + '/validation/'
    # folder_training = r'/home/fangx2/data/a_submit2/dataset_256/'
    # folder_validation = r'/home/fangx2/data/a_submit2/dataset_256/'
    # Set L2 penalty using weight_decay
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum)
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # resunet_checkpoint = torch.load('/home/fangx2/data/LIver_submit1/data1/tmp/concave_lr/resu_best_axial.pth.tar')
    # resunet_dict = resunet_checkpoint['state_dict']

    # model.load_state_dict(resunet_dict)

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

    weights = torch.Tensor([0.2, 1.2])
    criterion = CrossEntropyLoss2d(weights)
    # criterion = FocalLoss2d(weights)
    # criterion = DiceLoss()
    #criterion = HybridLoss2d(sobel, weights)
    #criterion = LoCeLoss2d(weights)
    
    if use_cuda:
        print('\n***** Training ResU-Net with GPU *****\n')
        model.cuda()
        criterion.cuda()

    blocksize = args.blocksize
    
    
    if view == 'axial' or view == 'sagittal' or view == 'coronal':
        composed = dl.get_composed_transform(blocksize, num_in_channels, view)
    else:

        print('The given view of <{}> is not supported!'.format(view))
        
    batchsize = args.batchsize

    dataset_train = dl.LiverCTDataset(folder_training,
                                      transform=composed)
    train_loader = dl.DataLoader(dataset_train,
                                 batch_size=args.batchsize,
                                 shuffle=True,
                                 num_workers=4,
                                 drop_last=False
                                )

    dataset_validation = dl.LiverCTDataset(folder_validation,
                                           transform=composed)
    val_loader = dl.DataLoader(dataset_validation,
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
        train_loss = train(train_loader, model, criterion, 
                           optimizer, epoch, verbose=True)
        train_history.append(train_loss)
        
        # Gradually reducing learning rate
        if epoch % 40 == 0:
            adjust_learning_rate(optimizer, gamma=0.99)

        # evaluate on validation set
        val_loss = validate(val_loader, model, criterion, epoch, verbose=True)
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
