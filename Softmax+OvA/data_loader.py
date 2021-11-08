import torch.utils.data as data
import h5py
import os
import numpy as np
import random
import matplotlib.pyplot as plt

def get_sigloader(data_path, train_class_num, save_path, train_transform=None, test_transform=None):
    data = h5py.File(data_path, 'r')
    x = np.array(data['X'], dtype='float32')
    y = np.array(data['Y'], dtype='long')[0]
    y -= 1
    classes = np.unique(y)
    cx = np.array(data['cX'], dtype='float32')
    cy = np.array(data['cY'], dtype='long')[0]
    cy -= 1

    idx = np.random.choice(classes, size=train_class_num, replace=False)
    np.savetxt(save_path+'/idx.txt', idx)
    class_map = np.array([train_class_num]*len(classes), dtype='long')
    for i in range(train_class_num):
        class_map[idx[i]] = i
    y = class_map[y]
    cy = class_map[cy]

    x = x[y<train_class_num,:,:,:]
    y = y[y<train_class_num]
    
    trainSigLoader = SigLoader(x, y, train_transform)
    testSigLoader = SigLoader(cx, cy, test_transform)
    return trainSigLoader, testSigLoader

def get_sigloader_fromlist(data_path, train_class_num, save_path, train_transform=None, test_transform=None):
    data = h5py.File(data_path, 'r')
    x = np.array(data['X'], dtype='float32')
    y = np.array(data['Y'], dtype='long')[0]
    y -= 1
    classes = np.unique(y)
    cx = np.array(data['cX'], dtype='float32')
    cy = np.array(data['cY'], dtype='long')[0]
    cy -= 1

    idx = np.loadtxt(save_path+'/idx.txt', dtype='long')
    class_map = np.array([train_class_num]*len(classes), dtype='long')
    for i in range(train_class_num):
        class_map[idx[i]] = i
    y = class_map[y]
    cy = class_map[cy]

    x = x[y<train_class_num,:,:,:]
    y = y[y<train_class_num]
    
    trainSigLoader = SigLoader(x, y, train_transform)
    testSigLoader = SigLoader(cx, cy, test_transform)
    return trainSigLoader, testSigLoader

class SigLoader(data.Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        # c, cnum = np.unique(y, return_counts=True)
        # print(cnum)
        self.transform = transform

    def __getitem__(self, item):
        imgs = self.x[item,:,:,:]
        labels = self.y[item]
        # plt.plot(imgs[0,:])
        if self.transform is not None:
            imgs = self.transform(imgs)
        # plt.plot(imgs[1,:])
        # plt.show()
        return imgs, int(labels)

    def __len__(self):
        return len(self.y)

