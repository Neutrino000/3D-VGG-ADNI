from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import scipy.io as io
import torch
#import torchvision
#from torchvision import transforms
#from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import nibabel as nib

class Mydataset(Dataset):
    def __init__(self, txtpath, nii_path, feature_map, scale, group):
        self.nii_path = nii_path
        self.scale = scale
        self.feature_map = feature_map
        self.scale = scale
        if group=='single':
            self.group = 'AR'
        else:
            self.group = 'ARglobal'
        imgs = []
        datainfo = open(txtpath, 'r')
        for line in datainfo:
            line = line.strip('/n')
            words = line.split()
            imgs.append((words[0], words[1]))
        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        sub, label = self.imgs[index]
        if self.feature_map=='all':
            fea_list = ['ALFF', 'fALFF', 'ReHo', 'VMHC']
            aa = np.zeros([4, 61, 61, 61])
            for i in range(4):
                pic = np.array(nib.load(self.nii_path+fea_list[i]+'_FunImg'+self.group+'CW/'+str(label)+self.scale+fea_list[i]+'Map_'+sub+'.nii').get_fdata()).swapaxes(0, -1).swapaxes(1, 2).swapaxes(1, -1)
                aa[i, :, :, :] = (pic[:, 5:66, :]-np.min(pic[:, 5:66, :]))/(np.max(pic[:, 5:66, :])-np.min(pic[:, 5:66, :]))
        else:
            aa = np.zeros([1, 61, 61, 61])
            pic = np.array(nib.load(self.nii_path+self.feature_map+'_FunImg'+self.group+'CW/'+str(label)+self.scale+self.feature_map+'Map_'+sub+'.nii').get_fdata()).swapaxes(0, -1).swapaxes(1, 2).swapaxes(1, -1)
            aa[0, :, :, :] = (pic[:, 5:66, :] - np.min(pic[:, 5:66, :])) / (np.max(pic[:, 5:66, :]) - np.min(pic[:, 5:66, :]))
        return aa, int(label)

# aaaa = Mydataset('train.txt', '/Volumes/yoshida/ADNI_ALFF_dataset/', 'all', '_z', 'global')
# bbbb = DataLoader(aaaa, batch_size=3, shuffle=True, num_workers=2)
# for pic, label in bbbb:
#     print(pic.shape)
