import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import sys
from VGGgap_3D import VGGgap
import nibabel as nib
import scipy.io as io
from cams import Cams
import cv2

def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())


def farward_hook(module, input, output):
    fmap_block.append(output)

def comp_class_vec(output_vec, index=None):
    if index is None:
        index = np.argmax(output_vec.cpu().data.numpy())
    else:
        index = np.array(index)
    index = index[:, np.newaxis]
    #print(index.shape)
    index = torch.from_numpy(index)
    one_hot = torch.zeros(4, 2).scatter_(1, index, 1).cuda()
    one_hot.requires_grad = True
    class_vec = torch.sum(one_hot * output)
    return class_vec

def show_cam_on_image(img, cam, out_dir, class1, output, name):
    img1 = img
    cam1 = cam
    ccam = cam1 + 0.2 * img1 / np.max(img1)
    ccam = (ccam - np.min(ccam)) / (np.max(ccam) - np.min(ccam))
    cam1 = cv2.applyColorMap(np.rot90(np.rot90(np.uint8(255 * ccam))), cv2.COLORMAP_JET)
    io.savemat(out_dir+'cam_'+str(class1)+name+'.mat', {'cam'+name: cam})
    cv2.imwrite(out_dir+"cam_"+str(class1)+'_'+str(output[class1])+name+".jpg", cam1)


if __name__=='__main__':
    num = int(sys.argv[1])
    class1 = int(sys.argv[2])
    methods = int(sys.argv[3])
    num_model = int(sys.argv[4])

    batch_size = 4
    feat_list = ['ALFF', 'fALFF', 'ReHo', 'VMHC']
    MNI152 = '/media/share/Member/song/ALFF/ADNI_ALFF/AAL_61x73x61_YCG.nii'
    mni = nib.load(MNI152).get_fdata()
    resnet = VGGgap(batch_size, 4, 16, 2, 0.8).cuda()
    resnet.load_state_dict(torch.load('/media/share/Member/song/featuremap_cnn/ADNI/all/VGGgap_normal_single/model/model_9.pkl'))
    resnet.eval()

    cam_list = ['gradcam', 'gradcampp', 'eigencam', 'eigengradcam', 'bas_cam']
    test_dir = '/media/share/Member/song/ALFF/ADNI_ALFF/ADNI_ALFF_dataset/'
#    test_list = os.listdir(test_dir)
    txt_path = '/media/share/Member/song/featuremap_cnn/ADNI/all/VGGgap_normal_single/code/random_split/test_9.txt'
    imgs = []
    datainfo = open(txt_path, 'r')
    for line in datainfo:
        line = line.strip('/n')
        words = line.split()
        imgs.append((words[0], words[1]))
    feat_list = ['ALFF', 'fALFF', 'ReHo', 'VMHC']
    rs_batch = np.zeros([batch_size, 4, 61, 61, 61])
    label_batch = np.zeros([batch_size, ])
    for i in range(batch_size):
        sub, label = imgs[batch_size * num + i]
        for j in range(4):
            pic = np.array(np.array(nib.load(test_dir + feat_list[j] + '_FunImgARCW/' + str(label) + '_' + feat_list[j] + 'Map_' + sub + '.nii').get_fdata())).swapaxes(0, -1).swapaxes(1, 2).swapaxes(1, -1)
            rs_batch[i, j, :, :, :] = (pic[:, 5:66, :] - np.min(pic[:, 5:66, :])) / (np.max(pic[:, 5:66, :]) - np.min(pic[:, 5:66, :]))
            label_batch[i,] = label

    label_name = label_batch.reshape([batch_size])
    rs_batch = Variable(torch.from_numpy(rs_batch)).float().cuda()
    label_batch = Variable(torch.from_numpy(label_batch)).long().cuda()
    fmap_block = list()
    grad_block = list()
    resnet.vgg_blok3.conv3.register_forward_hook(farward_hook)
    resnet.vgg_blok3.conv3.register_full_backward_hook(backward_hook)

    output = resnet(rs_batch)
    #    print(resnet.state_dict()['gap_fc.weight'].shape)
    if class1 == 0:
        gap_weight = resnet.state_dict()['gap_fc.weight'].detach().cpu().data.numpy()[0, :]
        label_batch1 = Variable(torch.from_numpy(np.zeros([batch_size, ]))).long()
    else:
        gap_weight = resnet.state_dict()['gap_fc.weight'].detach().cpu().data.numpy()[1, :]
        label_batch1 = Variable(torch.from_numpy(np.ones([batch_size, ]))).long()

    class_loss = comp_class_vec(output, label_batch1.cpu())
    class_loss.backward()

    for j in range(batch_size):
        # class_loss = comp_class_vec(output, label_batch1.cpu())
        # class_loss.backward()
        if torch.argmax(output[j]) == label_batch[j,]:
            cam_dir = '/media/share/Member/song/featuremap_cnn/ADNI/all/VGGgap_normal_single/cam_code/cam_test/cam_correct/' + cam_list[methods] + '/' + str(imgs[batch_size * num + j][1]) + '_' + str(imgs[batch_size * num + j][0]) + '/'
        else:
            cam_dir = '/media/share/Member/song/featuremap_cnn/ADNI/all/VGGgap_normal_single/cam_code/cam_test/cam_incorrect/' + cam_list[methods] + '/' + str(imgs[batch_size * num + j][1]) + '_' + str(imgs[batch_size * num + j][0]) + '/'

        # class_loss = comp_class_vec(output, label_batch.cpu())
        # class_loss.backward()
        grads = grad_block[0][j].cpu().data.numpy().squeeze()
        fmap = fmap_block[0][j].cpu().data.numpy().squeeze()
        cams1 = Cams(grads=grads, fmap=fmap, gap_weight = gap_weight)
        # cams1 = Cams(grads=grads, fmap=fmap)

        cmd = 'mkdir -p ' + cam_dir
        os.system(cmd)

        if methods == 0:
            cam1_x, cam1_y, cam1_z, cam_all = cams1.gradcam()
        elif methods == 1:
            cam1_x, cam1_y, cam1_z, cam_all = cams1.gradcampp()
        elif methods == 2:
            cam1_x, cam1_y, cam1_z, cam_all = cams1.eigencam()
        elif methods == 3:
            cam1_x, cam1_y, cam1_z, cam_all = cams1.eigengradcam()
        elif methods == 4:
            cam1_x, cam1_y, cam1_z, cam_all = cams1.bas_cam()

        back = np.zeros([61, 61, 61])
        back = mni[:, 5:66, :].swapaxes(0, -1).swapaxes(1, 2).swapaxes(1, -1)
        back_x = cv2.resize(back[30, :, :], (224, 224))
        #    back_x = cv2.applyColorMap(np.uint8(255*(back_x/np.max(back_x))), cv2.COLORMAP_BONE)
        back_y = cv2.resize(back[:, 30, :], (224, 224))
        #    back_y = cv2.applyColorMap(np.uint8(255*(back_y/np.max(back_y))), cv2.COLORMAP_BONE)
        back_z = cv2.resize(back[:, :, 30], (224, 224))
        #    back_z = cv2.applyColorMap(np.uint8(255*(back_z/np.max(back_z))), cv2.COLORMAP_BONE)
        output1 = output.cpu().data.numpy().squeeze()
        show_cam_on_image(img=back_x, cam=cam1_x, out_dir=cam_dir, class1=class1, output=output1[j], name='_x')
        show_cam_on_image(img=back_y, cam=cam1_y, out_dir=cam_dir, class1=class1, output=output1[j], name='_y')
        show_cam_on_image(img=back_z, cam=cam1_z, out_dir=cam_dir, class1=class1, output=output1[j], name='_z')
        io.savemat(cam_dir+'cam_all_'+str(class1)+'.mat', {'cam': cam_all})

