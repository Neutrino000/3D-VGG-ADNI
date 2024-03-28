import numpy as np
import cv2
import os

class Cams:
    def __init__(self, grads, fmap):
        self.grads = grads
        self.fmap = fmap
        #self.gap_weight = gap_weight

    #def bas_cam(self):
        #self.cam = np.zeros(self.fmap.shape[1:], dtype=np.float32)
        #for i in range(np.array(self.gap_weight).shape[0]):
            #self.cam += self.gap_weight[i]*self.fmap[i, :, :, :]
        #self.cam_x = cv2.resize(self.cam[int(self.cam.shape[0]/2), :, :], (224, 224))
        #self.cam_x = (self.cam_x - np.min(self.cam_x))/np.max(self.cam_x)
        #self.cam_y = cv2.resize(self.cam[:, int(self.cam.shape[1]/2), :], (224, 224))
        #self.cam_y = (self.cam_y - np.min(self.cam_y)) / np.max(self.cam_y)
        #self.cam_z = cv2.resize(self.cam[:, :, int(self.cam.shape[2]/2)], (224, 224))
        #self.cam_z = (self.cam_z - np.min(self.cam_z)) / np.max(self.cam_z)
        #return self.cam_x, self.cam_y, self.cam_z
        

    def gradcam(self):
        self.cam = np.zeros(self.fmap.shape[1:], dtype=np.float32)
        weights = np.mean(self.grads, axis=(1, 2, 3))
        for i, w in enumerate(weights):
            self.cam += w * self.fmap[i, :, :, :]
        #self.cam = np.maximum(self.cam, 0)
        self.cam_x = cv2.resize(self.cam[int(self.cam.shape[0]/2), :, :], (224, 224))
        self.cam_x = (self.cam_x - np.min(self.cam_x))/(np.max(self.cam_x)-np.min(self.cam_x))
        self.cam_y = cv2.resize(self.cam[:, int(self.cam.shape[1]/2), :], (224, 224))
        self.cam_y = (self.cam_y - np.min(self.cam_y)) / (np.max(self.cam_y)-np.min(self.cam_y))
        self.cam_z = cv2.resize(self.cam[:, :, int(self.cam.shape[2]/2)], (224, 224))
        self.cam_z = (self.cam_z - np.min(self.cam_z)) / (np.max(self.cam_z)-np.min(self.cam_z))
        return self.cam_x, self.cam_y, self.cam_z, weights

    def gradcampp(self):
        grads_power_2 = self.grads ** 2
        grads_power_3 = self.grads ** 3
        sum_activations = np.sum(self.fmap, axis=(1, 2, 3))
        eps = 0.000001
        aij = grads_power_2 / (2 * grads_power_2 + sum_activations[:, None, None, None] * grads_power_3 + eps)
        aij = np.where(self.grads != 0.0, aij, 1)
        #weights = np.maximum(self.grads, 0) * aij
        weights = np.multiply(np.maximum(self.grads, 0), aij)
        weights = np.sum(weights, axis=(1, 2, 3))
        self.cam = np.zeros(self.fmap.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            self.cam += w * self.fmap[i, :, :, :]

        #self.cam = np.maximum(self.cam, 0)
        self.cam_x = cv2.resize(self.cam[int(self.cam.shape[0]/2), :, :], (224, 224))
        self.cam_x = (self.cam_x - np.min(self.cam_x))/(np.max(self.cam_x)-np.min(self.cam_x))
        self.cam_y = cv2.resize(self.cam[:, int(self.cam.shape[1]/2), :], (224, 224))
        self.cam_y = (self.cam_y - np.min(self.cam_y)) / (np.max(self.cam_y)-np.min(self.cam_y))
        self.cam_z = cv2.resize(self.cam[:, :, int(self.cam.shape[2]/2)], (224, 224))
        self.cam_z = (self.cam_z - np.min(self.cam_z)) / (np.max(self.cam_z)-np.min(self.cam_z))
        return self.cam_x, self.cam_y, self.cam_z

    def eigencam(self):
        fmap = self.fmap.squeeze()
        fmap[np.isnan(fmap)] = 0
        reshaped_fmap = fmap.reshape(fmap.shape[0], -1).transpose()
        reshaped_fmap = reshaped_fmap - reshaped_fmap.mean(axis=0)
        U, S, VT = np.linalg.svd(reshaped_fmap, full_matrices=True)
        self.cam = reshaped_fmap @ VT[0, :]
        self.cam = self.cam.reshape(fmap.shape[1:])
        self.cam_x = cv2.resize(self.cam[int(self.cam.shape[0] / 2), :, :], (224, 224))
        self.cam_x = (self.cam_x - np.min(self.cam_x)) / (np.max(self.cam_x)-np.min(self.cam_x))
        self.cam_y = cv2.resize(self.cam[:, int(self.cam.shape[1] / 2), :], (224, 224))
        self.cam_y = (self.cam_y - np.min(self.cam_y)) / (np.max(self.cam_y)-np.min(self.cam_y))
        self.cam_z = cv2.resize(self.cam[:, :, int(self.cam.shape[2] / 2)], (224, 224))
        self.cam_z = (self.cam_z - np.min(self.cam_z)) / (np.max(self.cam_z)-np.min(self.cam_z))
        return self.cam_x, self.cam_y, self.cam_z



    def eigengradcam(self):
        fmap = self.grads.squeeze() * self.fmap.squeeze()
        fmap[np.isnan(fmap)] = 0
        reshaped_fmap = fmap.reshape(fmap.shape[0], -1).transpose()
        reshaped_fmap = reshaped_fmap - reshaped_fmap.mean(axis=0)
        U, S, VT = np.linalg.svd(reshaped_fmap, full_matrices=True)
        self.cam = reshaped_fmap @ VT[0, :]
        self.cam = self.cam.reshape(fmap.shape[1:])
        self.cam_x = cv2.resize(self.cam[int(self.cam.shape[0] / 2), :, :], (224, 224))
        self.cam_x = (self.cam_x - np.min(self.cam_x)) / (np.max(self.cam_x)-np.min(self.cam_x))
        self.cam_y = cv2.resize(self.cam[:, int(self.cam.shape[1] / 2), :], (224, 224))
        self.cam_y = (self.cam_y - np.min(self.cam_y)) / (np.max(self.cam_y)-np.min(self.cam_y))
        self.cam_z = cv2.resize(self.cam[:, :, int(self.cam.shape[2] / 2)], (224, 224))
        self.cam_z = (self.cam_z - np.min(self.cam_z)) / (np.max(self.cam_z)-np.min(self.cam_z))
        return self.cam_x, self.cam_y, self.cam_z
