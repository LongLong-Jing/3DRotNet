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
import input_data
import numpy as np
import math
import os
import build_model

# Training settings  for image convolution

parser = argparse.ArgumentParser(description='Image Reconstruction')

parser.add_argument('--batch_size', type=int, default=40, metavar='N',
                    help='input batch size for training (default: 40)')

parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')

parser.add_argument('--iteration', type=int, default=40000, metavar='N',
                    help='number of iterations to train (default: 40000)')

parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')

parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')

parser.add_argument('--weight_decay', type=float, default=0.005, metavar='M',
                    help='weight_decay (default: 0.005)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--save', type=str,  default='models/',
                    help='path to save the final model')

parser.add_argument('--log', type=str,  default='log/',
                    help='path to the log information')

def training(args):

    trainF = open(os.path.join(args.log, 'train.csv'), 'w')

    netG = build_model.netG().cuda(2)
    netD = build_model.netD().cuda(2)

    netG.train(True)
    netD.train(True)

    netG_optimizer = optim.Adam(netG.parameters(),lr=args.lr, betas=(0.5, 0.999), eps=1e-08, weight_decay=0)
    netD_optimizer = optim.Adam(netD.parameters(),lr=args.lr, betas=(0.5, 0.999), eps=1e-08, weight_decay=0)

    #baseline
    #pixel_criterion = nn.MSELoss().cuda(2)
    gan_criterion = nn.BCELoss().cuda(2)

    for step_index in range(args.iteration):

        train_image,train_label = input_data.ReadClip('list/ImageNet_Train.lst', args.batch_size, clip_length=1)

        train_image = np.transpose(train_image,(0,3,1,2));
        train_label = np.transpose(train_label,(0,3,1,2));

        train_image = torch.from_numpy(train_image)
        train_label = torch.from_numpy(train_label)

        real_label = torch.FloatTensor(args.batch_size).cuda(2)
        fake_label = torch.FloatTensor(args.batch_size).cuda(2)
        real_label, fake_label  = Variable(real_label), Variable(fake_label)
        real_label.data.fill_(1.0)
        fake_label.data.fill_(0.0)

        train_image, train_label = train_image.cuda(2), train_label.cuda(2)
        train_image, train_label = Variable(train_image), Variable(train_label)

        # optimize network D
        netD.zero_grad()

        real_output = netD(train_image)
        errD_real = gan_criterion(real_output,real_label)
        D_x = real_output.data.mean()
        fake = netG(train_image)

        fake_output = netD(fake.detach())
        errD_fake = gan_criterion(fake_output,fake_label)
        D_G_z1 = fake_output.data.mean()
        errD = (errD_real + errD_fake)*0.5
        errD.backward()
        netD_optimizer.step()

        # optimize network G
        netG.zero_grad()

        fake_output = netD(fake)
        #errG_L2 = pixel_criterion(fake,train_label)
        errG_gan = gan_criterion(fake_output, fake_label)
        D_G_z2 = fake_output.data.mean()
        errG = errG_gan*0.003
        errG.backward()
        netG_optimizer.step()

        if (step_index%10000) == 0:
            lr = args.lr * (0.1 ** (step_index // 10000))
            for param_group in netG_optimizer.param_groups:
                param_group['lr'] = lr
            for param_group in netD_optimizer.param_groups:
                param_group['lr'] = lr
        #accuracy
        if (step_index%50) ==0:
            print('Step: {},  G_loss: {},  D_loss: {},  D(x): {},  D(G(Z1)): {},  D(G(Z1)): {}\n'.format(step_index, errG.data[0], errD.data[0],D_x,D_G_z1,D_G_z2))
        trainF.write('Step: {},  G_loss: {},  D_loss: {},  D(x): {},  D(G(Z1)): {},  D(G(Z1)): {}\n'.format(step_index, errG.data[0], errD.data[0],D_x,D_G_z1,D_G_z2))
        if((step_index+1)%5000) ==0:
            print('----------------- Save The Network ------------------------\n')
            with open(args.save + str(step_index)+'netG', 'wb') as f:
                torch.save(netG, f)
            with open(args.save + str(step_index)+'netD', 'wb') as f:
                torch.save(netD, f)
    trainF.close()

def main():
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    training(args)

if __name__ == '__main__':
    main()
