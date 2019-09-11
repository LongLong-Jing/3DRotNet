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

parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 32)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--per_print', type=int,  default=1,
                    help='how many iterations to print the loss and accuracy')

parser.add_argument('--resume', type=str,  default='./models/80000net.pkl',
                    help='load the pre-trained model')

parser.add_argument('--resume_index', type=int,  default=0,
                    help='load the pre-trained model')

parser.add_argument('--log', type=str,  default='log/',
                    help='path to the log information')
parser.add_argument('--gpu_id', type=str,  default='0',
                    help='GPU used to train the network')

def training(args):

    print('load models from:  ' + args.resume)
    net = torch.load(args.resume).cuda()
    
    #net = torch.nn.DataParallel(net)
    #net = net.to('cuda')


    criterion = nn.CrossEntropyLoss()

    data_path = '../../Dataset/'
    dst = Video_Dataloader(root = data_path, dataset = 'MITS', is_train = True, is_transform=True)
    print('length of the dataset', len(dst))
    start = time.time()
    trainloader = torch.utils.data.DataLoader(dst, batch_size=args.batch_size,shuffle=True,num_workers=24)

    step_index = 0
    start_time = time.time()
    sum_sub = np.empty((1,4))
    sum_acc = 0
    video_cnt = 1.0
    for epoch in range(1):
        for i, data in enumerate(trainloader):
            clip_one, clip_two, clip_three, clip_four, target_one, target_two, target_three, target_four = data

            clip = torch.cat((clip_one, clip_two, clip_three, clip_four), dim=0)
            target = torch.cat((target_one, target_two, target_three, target_four), dim=0)
            clip = Variable(clip).cuda()
            target = Variable(target).cuda()
            
            pred = net(clip)
            #print(pred)
            #print(target)
	    acc, predict = calculate_accuracy(pred,target[:,0])
            target = target[:,0]
	    print('target    ',target)
            print('predict   ',predict)
            for index in range(4):
		if(target[index]== predict[0,index]):
                    sum_sub[0,index] = sum_sub[0,index]+1
            print(sum_sub/(video_cnt+1.0))
   
	    sum_acc += acc
            #print(i, acc, sum_acc/(video_cnt+1.0))
            video_cnt = video_cnt + 1

def main():
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    training(args)

if __name__ == '__main__':
    main()
