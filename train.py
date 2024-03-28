from config import args
from VGG_3D import VGG
from VGGgap_3D import VGGgap
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import os
import sys
import nibabel as nib
from data_make import Mydataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from random_split_ADNI import random_ge_ad
from random_split_COBRE import random_ge_sz

if __name__ == '__main__':
    if args.feature_map=='all':
        fm = 4
    else:
        fm = 1
    if args.scale == 'normal':
        scale_1 = '_'
    else:
        scale_1 = '_z'

    for num in range(args.kfold):
        network = locals()[args.model](args.batch_size, fm, args.channels, 2, args.dropout).cuda()
        optimizer = optim.Adam(network.parameters(), lr=args.lr, eps=0.1, weight_decay=args.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        kind_dir = '/media/share/Member/song/featuremap_cnn/'+args.dataset+'/'+args.feature_map+'/'+args.model+'_'+args.scale+'_'+args.group+'/'
        dataset_dir = '/media/share/Member/song/ALFF/'+args.dataset+'_ALFF/'+args.dataset+'_ALFF_dataset/'

        txt_train_path = kind_dir+'code/random_split/train_list_run_'+str(args.run_time)+'_k_'+str(num)+'_lr_'+str(args.lr)+'_ch_'+str(args.channels)+'_dp_'+str(int(args.dropout*10))+'.txt'
        txt_test_path = kind_dir+'code/random_split/test_list_run_'+str(args.run_time)+'_k_'+str(num)+'_lr_'+str(args.lr)+'_ch_'+str(args.channels)+'_dp_'+str(int(args.dropout*10))+'.txt'
        if args.dataset=='ADNI':
            random_ge_ad(dataset_dir, txt_train_path, txt_test_path)
        else:
            random_ge_sz(dataset_dir, txt_train_path, txt_test_path)

        train_data = Mydataset(txt_train_path, dataset_dir, args.feature_map, scale_1, args.group)
        test_data = Mydataset(txt_test_path, dataset_dir, args.feature_map, scale_1, args.group)
        test_data_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=1)

        loss_train = []
        loss_test = []
        acc_train = []
        acc_test = []
        acc_compare = 0
        processing = tqdm(range(args.epochs))

        for epoch in processing:
            num_batch = 0
            acc = 0
            loss_list = []
            network.train()
            train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=1)
            for pic, label in train_data_loader:
                optimizer.zero_grad()
                output = network(pic.float().cuda())
                loss = criterion(output, label.cuda())
                loss.backward()
                optimizer.step()
                loss_list.append(loss.cpu().detach().numpy())
                num_batch+=1
                for i in range(args.batch_size):
                    if torch.argmax(output[i, :])==label[i,]:
                        acc+=1
                acc_inter = acc/(num_batch*args.batch_size)
                if epoch>0:
                    processing.set_description('Epoch %d, Train loss %.4f, Train acc %.3f, Test loss %.4f, Test acc %.3f' %(epoch+1, np.mean(loss_list), acc_inter, loss_test[epoch-1], acc_test[epoch-1]))

            acc = acc/(len(train_data_loader)*args.batch_size)
            acc_train.append(acc)
            np.savetxt(kind_dir+'loss/acc_train_run_'+str(args.run_time)+'_k_'+str(num)+'_lr_'+str(args.lr)+'_ch_'+str(args.channels)+'_dp_'+str(int(args.dropout*10))+'.txt', acc_train)
            loss_train.append(np.mean(loss_list))
            np.savetxt(kind_dir+'loss/loss_train_run_'+str(args.run_time)+'_k_'+str(num)+'_lr_'+str(args.lr)+'_ch_'+str(args.channels)+'_dp_'+str(int(args.dropout*10))+'.txt', loss_train)

            scheduler.step()

            acc_t = 0
            loss_list = []
            network.eval()
            with torch.no_grad():
                for pic, label in test_data_loader:
                    output = network(pic.float().cuda())
                    loss = criterion(output, label.cuda())
                    loss_list.append(loss.cpu().detach().numpy())
                    for i in range(args.batch_size):
                        if torch.argmax(output[i, :])==label[i,]:
                            acc_t+=1
                acc_t = acc_t/(len(test_data_loader)*args.batch_size)
                acc_test.append(acc_t)
                np.savetxt(kind_dir + 'loss/acc_test_run_' + str(args.run_time) + '_k_' + str(num) + '_lr_' + str(
                    args.lr) + '_ch_' + str(args.channels) + '_dp_' + str(int(args.dropout * 10)) + '.txt', acc_test)
                loss_test.append(np.mean(loss_list))
                np.savetxt(kind_dir + 'loss/loss_test_run_' + str(args.run_time) + '_k_' + str(num) + '_lr_' + str(
                    args.lr) + '_ch_' + str(args.channels) + '_dp_' + str(int(args.dropout * 10)) + '.txt', loss_test)

            if acc_t >= acc_compare:
                acc_compare = acc_t
                torch.save(network.state_dict(), kind_dir+'model/model_'+str(num)+'.pkl')

