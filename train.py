from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from numpy.random import normal
from numpy.linalg import svd
from math import sqrt
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import random
from torch.utils import data
import torchvision
from resnet import resnet34, resnet10, resnet18, resnet50
from Video_dataloader import Video_Dataloader
import time
from utils import calculate_accuracy
from torch.optim import lr_scheduler


parser = argparse.ArgumentParser(description='3DRotNet for the self-supervised learning of video features')

parser.add_argument('--pre_trained', type=bool, default=False, metavar='Pre_trained',
                    help='input batch size for training (default: False)')

parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 96)')

parser.add_argument('--epoch', type=int, default=100, metavar='Epoch Number',
                    help='number of iterations to train (default: 100)')

parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.005)')

parser.add_argument('--momentum', type=float, default=0.9, metavar='momentum',
                    help='learning rate (default: 0.9)')

parser.add_argument('--weight_decay', type=float, default=1e-3, metavar='weight_decay',
                    help='learning rate (default: 1e-3)')

parser.add_argument('--dampening', type=float, default=0.9, metavar='dampening',
                    help='learning rate (default: 1e-3)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--per_save', type=int,  default=4000,
                    help='how many iterations to save the model')

parser.add_argument('--per_print', type=int,  default=10,
                    help='how many iterations to print the loss and accuracy')

parser.add_argument('--resume', type=str,  default='models-rgb/116000net.pkl',
                    help='load the pre-trained model')

parser.add_argument('--save', type=str,  default='models-rgb/',
                    help='path to save the final model')

parser.add_argument('--log', type=str,  default='log/',
                    help='path to the log information')

# CUDA_VISIBLE_DEVICES=1 python train

def training(args):

    if args.resume:
        print('load models from:  ' + args.resume)
        print('learning rate: ' + str(args.lr))
        net = torch.load(args.resume).cuda()
    else:
        net = resnet18(num_classes = 4, shortcut_type = 'A', sample_size = 112, sample_duration = 16).cuda()
    
    net = net.cuda()

    optimizer = optim.SGD(
            net.parameters(),
            lr=0.0001,
            momentum=args.momentum,
            weight_decay=args.weight_decay)

    net.train(True)

    criterion = nn.CrossEntropyLoss()

    data_path = '../../Dataset/'
    dst = Video_Dataloader(root = data_path, dataset = 'KITS', is_train = True, is_transform=True)
    print('length of the dataset', len(dst))
    start = time.time()
    trainloader = torch.utils.data.DataLoader(dst, batch_size=args.batch_size,shuffle=True,num_workers=12)

    step_index = 116000
    start_time = time.time()
    for epoch in range(args.epoch):
        for i, data in enumerate(trainloader):
            clip_one, clip_two, clip_three, clip_four, target_one, target_two, target_three, target_four  = data
            clip_one = Variable(clip_one).cuda()
            clip_two = Variable(clip_two).cuda()
            clip_three = Variable(clip_three).cuda()
            clip_four = Variable(clip_four).cuda()
            target_one = Variable(target_one).cuda()
            target_two = Variable(target_two).cuda()
            target_three = Variable(target_three).cuda()
            target_four = Variable(target_four).cuda()

            clip = torch.cat((clip_one, clip_two, clip_three, clip_four), dim=0)
            target = torch.cat((target_one, target_two, target_three, target_four), dim=0)
            
            # print('----------------------------')
            # print(clip_one.size())
            # print(clip_two.size())
            print(clip.size())
            # print('----------------------------')
            
            
            net.zero_grad()
            pred = net(clip)
            # print(pred.size())
            # print(target.size())
            loss = criterion(pred,target[:,0])
            loss.backward()
            optimizer.step()

            acc, pred_cls = calculate_accuracy(pred,target[:,0])


            if (step_index%24000) == 0:
                lr = args.lr * (0.1 ** (step_index // 24000))
                print('------------    LR    ----------------')
                print(lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            if step_index % args.per_print == 0:
                print('[%d][%d][%d] accuracy: %f  loss: %f  time: %f ' % (epoch, i, step_index, acc, loss.item(), time.time() - start_time))
                start_time = time.time()
            if((step_index+1) % args.per_save) ==0:
                print('----------------- Save The Network ------------------------\n')
                with open(args.save + str(step_index+1)+'net.pkl', 'wb') as f:
                    torch.save(net, f)
            step_index = step_index + 1

def main():
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    training(args)

if __name__ == '__main__':
    main()
