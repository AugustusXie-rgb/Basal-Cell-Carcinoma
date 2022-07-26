import os
import torch.utils.data as data
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from PIL import Image
import time
import copy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import newaxis

NUM_EPOCH = 5
batch_size = 64
device = torch.device('cuda:0')
NUMCLASS = 2

def default_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            # img = np.asarray(img)
            # # img = img[:,:,newaxis]
            # img_rgb=np.zeros((img.shape[0], img.shape[1], 3))
            # img_rgb[:,:,0] = img
            # img = Image.fromarray(np.uint8(img_rgb))
            return img.convert('RGB')

class CustomImageLoader(data.Dataset):
    def __init__(self, B_path, N_path, dataset='', data_transforms=None, loader=default_loader):
        im_list = []
        im_labels = []
        B_images = os.listdir(B_path)
        N_images = os.listdir(N_path)
        for img in B_images:
            im_list.append(os.path.join(B_path,img))
            im_labels.append(1)
        for img in N_images:
            im_list.append(os.path.join(N_path,img))
            im_labels.append(0)

        self.imgs = im_list
        self.labels = im_labels
        self.data_transforms = data_transforms
        self.loader = loader
        self.dataset = dataset

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        img_name = self.imgs[item]
        label = self.labels[item]
        img = self.loader(img_name)

        if self.data_transforms is not None:
            try:
                img = self.data_transforms[self.dataset](img)
            except:
                print('Cannot transform: {}'.format(img_name))
        return img, label

data_transforms={
    'Train':transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]),
    'Test':transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
}

image_datasets = {'Train' : CustomImageLoader(B_path='/home/bfl/XieJun/keras_resnet/dataset/comp/group_3/1cut_train/B/',
                                        N_path='/home/bfl/XieJun/keras_resnet/dataset/comp/group_3/1cut_train/N/',
                                        data_transforms=data_transforms,
                                        dataset='Train'),
                  'Test': CustomImageLoader(B_path='/home/bfl/XieJun/keras_resnet/dataset/comp/group_3/1cut_val1/B/',
                                            N_path='/home/bfl/XieJun/keras_resnet/dataset/comp/group_3/1cut_val1/N/',
                                            data_transforms=data_transforms,
                                            dataset='Test')}

dataloaders = {'Train': torch.utils.data.DataLoader(image_datasets['Train'],batch_size=batch_size, shuffle=True),
               'Test': torch.utils.data.DataLoader(image_datasets['Test'], batch_size=batch_size, shuffle=True)}

dataset_sizes = {'Train': len(image_datasets['Train']),
                 'Test': len(image_datasets['Test'])}

def train_model(model, crtiation, optimizer, schedular, num_epochs=NUM_EPOCH):
    begin_time = time.time()
    best_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    arr_acc = []

    for epoch in range(num_epochs):
        print("-*-" * 20)
        item_acc = []
        for phase in ['Train', "Test"]:
            if phase=='Train':
                schedular.step()
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_acc = 0.0

            for images, labels in dataloaders[phase]:
                images.to(device)
                labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='Train'):
                    opt = model(images.cuda())
                    _,pred = torch.max(opt, 1)
                    labels = labels.cuda()
                    loss = crtiation(opt, labels)
                    if phase=='Train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()*images.size(0)
                running_acc += torch.sum(pred==labels)


            epoch_loss = running_loss/dataset_sizes[phase]
            epoch_acc = running_acc.double()/dataset_sizes[phase]
            print('epoch={}, Phase={}, Loss={:.4f}, ACC:{:.4f}'.format(epoch, phase, epoch_loss, epoch_acc))

            item_acc.append(epoch_acc)

            if phase == 'Test' and epoch_acc>best_acc:
                best_acc=epoch_acc
                best_weights = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), './checkpoint/mil_pre_resnet/checkpoint-{}e-val_accuracy_{}'.format(epoch, best_acc))

        arr_acc.append(item_acc)
        # time_elapes = time.time() - begin_time

    print('Best Val ACC: {:}'.format(best_acc))

    model.load_state_dict(best_weights)
    return model, arr_acc

if __name__=='__main__':
    model_ft = models.resnet101(pretrained=False)
    num_fits = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_fits, NUMCLASS)
    model_ft = model_ft.to(device)
    model_ft.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10000, gamma=0.1)
    model_ft, arr_acc = train_model(model_ft, criterion, optimizer=optimizer_ft, schedular=exp_lr_scheduler, num_epochs=20)

# ll = np.array(arr_acc)
# plt.plot(ll[:,0])
# plt.plot(ll[0:1])


