import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import torchvision.models as models
from torchvision import transforms

train_transforms = transforms.Compose([
    # transforms.RandomCrop(224),
    # transforms.CenterCrop(224),

    # transforms.RandomHorizontalFlip(),
    # transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
test_transforms = transforms.Compose([
    # transforms.Resize(1024),
    # transforms.RandomCrop(224),
    # transforms.CenterCrop(224),
    # transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
#
# # 数据集加载方式设置
# train_data = BuyanDataset('D:/pengbo/dataset/buyan_cdx_cls/train_aug.txt', transform=train_transforms)
# test_data = BuyanDataset('D:/pengbo/dataset/buyan_cdx_cls/test_aug.txt', transform=test_transforms)
# print(train_data.__getitem__(0))
#
# class Dataloder():
#     def __init__(self, args):
#         self.trainloader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
#         self.testloader = DataLoader(dataset=test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=0)
#
#     def getloader(self):
#         return self.trainloader, self.testloader

class BuyanDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.data)
        return self.filelength

    def __getitem__(self, index):
        img = self.data[index]
        # img = Image.open(fn).convert('RGB')
        # if self.transform is not None:
        label = self.labels[index]
        img = self.transform(img)
        # print("hhhhh:label",label)
        return img, int(label)
# root = 'D:/pengbo/code/Federated-Learning-PyTorch-master/data/datesets/'
root = '/home/pengbo/code/Federated-Learning-PyTorch-master/data/datesets'
# train_data1 = np.load(root + '/buyan/buyan_c_data.npy')
# train_cls1 = np.load(root +'buyan/buyan_c_label.npy')
# train_data2 = np.load(root +'buyan/buyan_p_data.npy')
# train_cls2 = np.load(root +'buyan/buyan_p_label.npy')
# train_data3 = np.load(root +'buyan/buyan_v_data.npy')
# train_cls3 = np.load(root +'buyan/buyan_v_label.npy')
# train_data4 = np.load(root +'buyan/buyan_l_data.npy')
# train_cls4 = np.load(root +'buyan/buyan_l_label.npy')
# train_data5 = np.load(root +'buyan/buyan_n_data.npy')
# train_cls5 = np.load(root +'buyan/buyan_n_label.npy')

train_data_all = np.load(root + '/buyan/all_train_data.npy')
train_cls_all = np.load(root + '/buyan/all_train_label.npy')
test_data_all = np.load(root + '/buyan/all_test_data.npy')
test_cls_all = np.load(root + '/buyan/all_test_label.npy')


print(train_data_all.shape)
print(test_data_all.shape)
# test_cls
#
# train_data = np.concatenate((train_data1[:545],train_data2[:800],train_data3[:400],train_data4[:1200],train_data5[:200]),axis=0)
# train_cls = np.concatenate((train_cls1[:545],train_cls2[:800],train_cls3[:400],train_cls4[:1200],train_cls5[:200]),axis=0)
#
# test_data = np.concatenate((train_data1[545:],train_data2[800:],train_data3[400:],train_data4[1200:],train_data5[200:]),axis=0)
# test_cls = np.concatenate((train_cls1[545:],train_cls2[800:],train_cls3[400:],train_cls4[1200:],train_cls5[200:]),axis=0)

train_data = BuyanDataset(train_data_all, train_cls_all, transform=train_transforms)
test_data = BuyanDataset(test_data_all, test_cls_all, transform=test_transforms)

# train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=False)