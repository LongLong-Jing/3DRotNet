import os
import collections
import json
import torch
import torchvision
import torchvision.transforms as transforms
from videotransforms import video_transforms, volume_transforms
import numpy as np
import PIL.Image as Image
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from torch.utils import data
from PIL import Image
import os
import os.path
import cv2
import PIL
from PIL import Image
import h5py
import scipy
import random
from os import listdir
from os.path import isfile, join
import time
import h5py
import random

class Video_Dataloader(data.Dataset):
    def __init__(self, root, dataset, is_train,is_transform=True, img_size=112):
        self.root = root
        self.dataset = dataset
        self.is_train = is_train
        self.is_transform = is_transform
        self.nframes = 0
        self.video_transform_one = video_transforms.Compose(
            [
                video_transforms.RandomRotation(10),
                video_transforms.RandomCrop((112, 112)),
                video_transforms.RandomHorizontalFlip(),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize((0.4339, 0.4046, 0.3776), (0.151987, 0.14855, 0.1569))
            ]           
        )

        #########################################################################
        # load the image feature and image name 
        ##########################################################################
        if self.is_train:
            if self.dataset == 'MITS':
                vid_list = open('./MITS_split/MITS.lst')
                self.nframes = 90
            elif self.dataset == 'KITS':
                vid_list = open('./KITS_split/KITS.lst')
                self.nframes = 160
            else:
                print('no such dataset')
                return
            vid_list = list(vid_list)

            self.data_list = []

            for line in vid_list:
                line = line.strip('\r\n')
                # print(line)
                self.data_list.append(line)

    def __len__(self):
        #This should use the positive example
        return len(self.data_list)

    def __getitem__(self, index):
        # index is the postive example
        path = self.data_list[index]

        hf = h5py.File(path, 'r')
        key = hf.keys()
        data_obj  = hf.get(key[0])
        data_list = data_obj.items()
        np_data = data_obj.get(data_list[0][0])
        np_data = np.array(np_data, dtype = np.float)
        
        # print(np_data.shape)
        # Each video has around 100 frames
        start = random.randint(0,self.nframes-32)
        #This value is depends on the 
        # print(start)

        clipx = []
        index = start

        while index < start+32:
            data = np_data[index,:,:,:]
            clipx.append(data)
            index = index + 2

        clip_one = self.video_transform_one(clipx)
        # clip_one = clip_one[:,1:17,:,:] - clip_one[:,0:16,:,:]
        # #print(clip_one.size())
        clip_one = clip_one.numpy()
        clip_two = np.rot90(clip_one, k = 1 , axes = (2,3))
        clip_three = np.rot90(clip_one, k = 2 , axes = (2,3))
        clip_four = np.rot90(clip_one, k = 3 , axes = (2,3))

        clip_one = torch.from_numpy(clip_one.copy())
        clip_two = torch.from_numpy(clip_two.copy())
        clip_three = torch.from_numpy(clip_three.copy())
        clip_four = torch.from_numpy(clip_four.copy())

        target_one = torch.LongTensor(np.array([0]))
        target_two = torch.LongTensor(np.array([1]))
        target_three = torch.LongTensor(np.array([2]))
        target_four = torch.LongTensor(np.array([3]))

        # print('=====================')
        # print(clip_one.size())
        # print(clip_two.size())
        # print(clip_three.size())
        # print(clip_four.size())
        # print(target_one)
        # print(target_two)
        # print(target_three)
        return clip_one, clip_two, clip_three, clip_four, target_one, target_two, target_three, target_four

if __name__ == '__main__':
    data_path = '../../Dataset/'
    dst = Video_Dataloader(root = data_path, dataset = 'KITS', is_train = True, is_transform=True)
    print('length of the dataset', len(dst))
    start = time.time()
    trainloader = data.DataLoader(dst, batch_size=30,shuffle=True,num_workers=20)
    for i, data in enumerate(trainloader):
        clip_one, clip_two, clip_three, clip_four, target_one, target_tow, target_three, target_four  = data
        #print(img_clip.size())
        #print(img_clip)
        # print('--------------------------------------')
        print(i, time.time()-start)
        # print(clip_one.size())
        # print(clip_two.size())
        # print(target)
        # print('--------------------------------------')
        # print(i, time.time() - start)
        # start = time.time()
        # print(img_clip.size())
        # print(video_path)
        #print(cat_name)
        #print(vid_name)
        #print('---------------------------')
        # print(video_clip.size())
        # print(video_clip.size())
        # print(mask_clip.size())
        # # print(mask_clip)
        # # print(video_clip.size())
        #(48, 3, 16, 112, 112)
        # print(clip_one.size())
        # img_one = clip_one.numpy()[0,:,8,:,:]
        # img_two = clip_two.numpy()[0,:,8,:,:]
        # img_three = clip_three.numpy()[0,:,8,:,:]
        # img_four = clip_four.numpy()[0,:,8,:,:]

        # img_one = np.transpose(img_one,(1,2,0))
        # img_two = np.transpose(img_two,(1,2,0))
        # img_three = np.transpose(img_three,(1,2,0))
        # img_four = np.transpose(img_four,(1,2,0))

        # print(np.amin(img_one),np.amax(img_one))
        # print(np.amin(img_two),np.amax(img_two))
        # print(np.amin(img_three),np.amax(img_three))
        # print(np.amin(img_four),np.amax(img_four))
        # print(np.sum(imgs))
        # print(mask.dtype)
        # cv2.imwrite(str(i)+'.png',imgs)
        # cv2.imshow('img_one', img_one)
        # cv2.imshow('img_two',img_two)
        # cv2.imshow('img_three', img_three)
        # cv2.imshow('img_four',img_four)
        # cv2.waitKey(3000)
