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
from MITS_dataloader import MITS_Dataloader
import time
from utils import calculate_accuracy
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tempfile import TemporaryFile
import cv2

# Training settings  for image convolution

parser = argparse.ArgumentParser(description='Video Super Resolution With Generative Advresial Network')

parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 32)')

parser.add_argument('--iteration', type=int, default=40000, metavar='N',
                    help='number of iterations to train (default: 40000)')

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

def testing(args):

    predict_array = np.empty([1011,101])
    gt_label = np.empty([1011])

    net = torch.load('./models/4000net.pkl')

    net.eval()


    modules=list(net.children())[:-4]
    vis_model=nn.Sequential(*modules)
    print(vis_model)



    data_path = '/media/longlong/d93b8ec0-0f86-4d0d-ba5b-179406741e41/'
    dst = MITS_Dataloader(data_path, is_train = True, is_transform=True)
    print('length of the dataset', len(dst))
    start = time.time()
    trainloader = torch.utils.data.DataLoader(dst, batch_size=1,shuffle=True,num_workers=0)
    

    step_index = 0


    # prediction_labels = []
    # groundtruth_labels = []

    path = './feat/'

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

        video_clip = torch.cat((clip_one, clip_two, clip_three, clip_four), dim=0)
        label = torch.cat((target_one, target_two, target_three, target_four), dim=0)
            

        # label = label[:,0]
        # video_clip, label = Variable(video_clip).cuda(), Variable(label).cuda()
        
        pred = vis_model(video_clip)
        # print(time.time() - start_time)
        # print(pred.size())
        # print(video_clip.size())
        video_clip = video_clip.data.cpu().numpy()
        if not os.path.exists(path+str(i)):
            os.mkdir(path+str(i))
	print(path+str(i))
        
        for index in range(16):
            img = video_clip[0,:,index,:,:]
            # print(img.shape)
            img = (img - np.amin(img))/(np.amax(img) - np.amin(img))
            img = np.transpose(img,(1,2,0))
            img = cv2.resize(img,(200,200))
            # print(img.shape)
            cv2.imwrite(path+str(i) + '/' + str(index) + '.png', img*255)

        featuremap = pred.data.cpu().numpy()
        print(featuremap.shape)
        for index in range(4):
            feat = featuremap[0,:,index,:,:]
            # print(feat.shape)
            x = np.empty([14,14])
            x.fill(0)
            for h in range(128):
                # feat[h,:,:] = (feat[h,:,:] - np.amin(feat[h,:,:]))/(np.amax(feat[h,:,:])- np.amin(feat[h,:,:]))
                x = x + feat[h,:,:]
            x = (x - np.amin(x))/(np.amax(x) - np.amin(x))
            x = cv2.resize(x,(200,200))
            heat_img =  np.uint8(cm.jet(1 - x)*255)
            # print(heat_img.shape)
            #print(path+str(i) + '/' + str(index+30) + '.png')
            #print(np.amin(heat_img), np.amax(heat_img))
            cv2.imwrite(path+str(i) + '/' + str(index+30) + '.png',heat_img)
            # print('-------------------')
            # cv2.imshow('feat',heat_img)
            # cv2.waitKey(1000)



        step_index = step_index + 1
        # break



def main():
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    testing(args)

if __name__ == '__main__':
    main()






# from __future__ import print_function
# import argparse
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.autograd import Variable
# from numpy.random import normal
# from numpy.linalg import svd
# from math import sqrt
# import numpy as np
# import math
# import os
# import matplotlib.pyplot as plt
# import random
# from torch.utils import data
# import torchvision
# from resnet import resnet34, resnet10, resnet18, resnet50
# from MITS_dataloader import MITS_Dataloader
# import time
# from utils import calculate_accuracy
# from torch.optim import lr_scheduler
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import itertools
# from sklearn import svm, datasets
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix
# from tempfile import TemporaryFile
# import cv2

# # Training settings  for image convolution

# parser = argparse.ArgumentParser(description='Video Super Resolution With Generative Advresial Network')

# parser.add_argument('--batch_size', type=int, default=1, metavar='N',
#                     help='input batch size for training (default: 32)')

# parser.add_argument('--iteration', type=int, default=40000, metavar='N',
#                     help='number of iterations to train (default: 40000)')

# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='enables CUDA training')

# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                     help='random seed (default: 1)')

# parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                     help='how many batches to wait before logging training status')

# parser.add_argument('--save', type=str,  default='models/',
#                     help='path to save the final model')

# parser.add_argument('--log', type=str,  default='log/',
#                     help='path to the log information')

# def testing(args):

#     predict_array = np.empty([1011,101])
#     gt_label = np.empty([1011])

#     net = torch.load('./models/4000net.pkl')

#     net.eval()


#     modules=list(net.children())[:-5]
#     vis_model=nn.Sequential(*modules)
#     print(vis_model)



#     data_path = '/media/longlong/d93b8ec0-0f86-4d0d-ba5b-179406741e41/'
#     dst = MITS_Dataloader(data_path, is_train = True, is_transform=True)
#     print('length of the dataset', len(dst))
#     start = time.time()
#     trainloader = torch.utils.data.DataLoader(dst, batch_size=1,shuffle=True,num_workers=0)
    

#     step_index = 0


#     # prediction_labels = []
#     # groundtruth_labels = []

#     path = './feat/'

#     for i, data in enumerate(trainloader):
#         clip_one, clip_two, clip_three, clip_four, target_one, target_two, target_three, target_four  = data
#         clip_one = Variable(clip_one).cuda()
#         clip_two = Variable(clip_two).cuda()
#         clip_three = Variable(clip_three).cuda()
#         clip_four = Variable(clip_four).cuda()
#         target_one = Variable(target_one).cuda()
#         target_two = Variable(target_two).cuda()
#         target_three = Variable(target_three).cuda()
#         target_four = Variable(target_four).cuda()

#         video_clip = torch.cat((clip_one, clip_two, clip_three, clip_four), dim=0)
#         label = torch.cat((target_one, target_two, target_three, target_four), dim=0)
            

#         # label = label[:,0]
#         # video_clip, label = Variable(video_clip).cuda(), Variable(label).cuda()
        
#         pred = vis_model(video_clip)
#         # print(time.time() - start_time)
#         # print(pred.size())
#         # print(video_clip.size())
#         video_clip = video_clip.data.cpu().numpy()
#         if not os.path.exists(path+str(i)):
#             os.mkdir(path+str(i))
# 	print(path+str(i))
        
#         for index in range(16):
#             img = video_clip[0,:,index,:,:]
#             # print(img.shape)
#             img = (img - np.amin(img))/(np.amax(img) - np.amin(img))
#             img = np.transpose(img,(1,2,0))
#             img = cv2.resize(img,(200,200))
#             # print(img.shape)
#             cv2.imwrite(path+str(i) + '/' + str(index) + '.png', img*255)

#         featuremap = pred.data.cpu().numpy()
#         print(featuremap.shape)
#         for index in range(8):
#             feat = featuremap[0,:,index,:,:]
#             # print(feat.shape)
#             x = np.empty([28,28])
#             x.fill(0)
#             for h in range(64):
#                 # feat[h,:,:] = (feat[h,:,:] - np.amin(feat[h,:,:]))/(np.amax(feat[h,:,:])- np.amin(feat[h,:,:]))
#                 x = x + feat[h,:,:]
#             x = (x - np.amin(x))/(np.amax(x) - np.amin(x))
#             x = cv2.resize(x,(200,200))
#             heat_img =  np.uint8(cm.jet(1 - x)*255)
#             # print(heat_img.shape)
#             #print(path+str(i) + '/' + str(index+30) + '.png')
#             #print(np.amin(heat_img), np.amax(heat_img))
#             cv2.imwrite(path+str(i) + '/' + str(index+30) + '.png',heat_img)
#             # print('-------------------')
#             # cv2.imshow('feat',heat_img)
#             # cv2.waitKey(1000)


#         step_index = step_index + 1
#         # break



# def main():
#     args = parser.parse_args()
#     args.cuda = not args.no_cuda and torch.cuda.is_available()
#     torch.manual_seed(args.seed)
#     if args.cuda:
#         torch.cuda.manual_seed(args.seed)
#     testing(args)

# if __name__ == '__main__':
#     main()



