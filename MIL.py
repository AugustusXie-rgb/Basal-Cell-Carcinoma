import os
import datetime
import copy
import re
import yaml
import uuid
import warnings
import time
import inspect
import pickle

import numpy as np
import pandas as pd
from functools import partial, reduce
from random import shuffle
import random
from PIL import Image

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision.models import resnet, resnet101
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.datasets import MNIST
from tqdm.autonotebook import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn import metrics as mtx
from sklearn import model_selection as ms

from typing import List, Dict, Tuple

class Transform_data(Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data
    def __getitem__(self, index):
        tensor = self.data[index][0]
        if self.transform is not None:
            tensor = self.transform(tensor)
        return (tensor, self.data[index][1])
    def __len__(self):
        return len(self.data)

bag_indices = pickle.load(open("./MIL_data/train/bags", "rb"))
bag_labels = pickle.load(open("./MIL_data/train/bags_labels", "rb"))
feature_array = np.load('./MIL_data/train/feature_array.npy', allow_pickle=True)
bag_features = {kk: torch.Tensor(feature_array[inds]) for kk, inds in bag_indices.items()}

bag_t_indices = pickle.load(open("./MIL_data/val/bags", "rb"))
bag_t_labels = pickle.load(open("./MIL_data/val/bags_labels", "rb"))
feature_test_array = np.load('./MIL_data/val/feature_array.npy', allow_pickle=True)
bag_t_features = {kk: torch.Tensor(feature_test_array[inds]) for kk, inds in bag_t_indices.items()}

train_data = [(bag_features[i], bag_labels[i]) for i in range(len(bag_features))]

def pad_tensor(data:list, max_number_instance) -> list:
    new_data = []
    for bag_index in range(len(data)):
        tensor_size = len(data[bag_index][0])
        pad_size = max_number_instance - tensor_size
        p2d = (0, 0, 0, pad_size)
        padded = nn.functional.pad(data[bag_index][0], p2d, 'constant', 0)
        new_data.append((padded, data[bag_index][1]))
    return new_data

max_number_instance = 90
padded_train = pad_tensor(train_data, max_number_instance=max_number_instance)

test_data = [(bag_t_features[i], bag_t_labels[i]) for i in range(len(bag_t_features))]
padded_test = pad_tensor(test_data, max_number_instance=max_number_instance)

def get_data_loaders(train_data, test_data, train_batch_size, val_batch_size):
    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(test_data, batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader

train_batch_size = 1
val_batch_size = 1
train_loader, valid_loader = get_data_loaders(padded_train, padded_test, train_batch_size=train_batch_size, val_batch_size=val_batch_size)


class SoftMaxMeanSimple(torch.nn.Module):
    def __init__(self, n, n_inst, dim=0):
        super(SoftMaxMeanSimple, self).__init__()
        self.dim = dim
        self.gate = torch.nn.Softmax(dim=self.dim)
        self.mdl_instance_transform = nn.Sequential(
            nn.Linear(n, n_inst),
            nn.LeakyReLU(),
            nn.Linear(n_inst, n),
            nn.LeakyReLU,
        )

    def forward(self, x):
        z = self.mdl_instance_transform(x)
        if self.dim == 0:
            z = z.view((z.shape[0], 1)).sum(1)
        elif self.dim == 1:
            z = z.view((1, z.shape[1])).sum(0)
        gate_ = self.gate(z)
        res = torch.sum(x * gate_, self.dim)
        return res, gate_


class AttentionSoftMax(torch.nn.Module):
    def __init__(self, in_features=3, out_features=None):
        super(AttentionSoftMax, self).__init__()
        self.otherdim = ''
        if out_features is None:
            out_features = in_features
        self.layer_linear_tr = nn.Linear(in_features, out_features)
        self.activation = nn.LeakyReLU()
        self.layer_linear_query = nn.Linear(out_features, 1)

    def forward(self, x):
        keys = self.layer_linear_tr(x)
        keys = self.activation(keys)
        attention_map_raw = self.layer_linear_query(keys)[..., 0]
        attention_map = nn.Softmax(dim=-1)(attention_map_raw)
        result = torch.einsum(f'{self.otherdim}i,{self.otherdim}ij->{self.otherdim}j', attention_map, x)
        return result, attention_map


class NoisyAnd(torch.nn.Module):
    def __init__(self, a=10, dims=[1, 2]):
        super(NoisyAnd, self).__init__()
        #         self.output_dim = output_dim
        self.a = a
        self.b = torch.nn.Parameter(torch.tensor(0.01))
        self.dims = dims
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #         h_relu = self.linear1(x).clamp(min=0)
        mean = torch.mean(x, self.dims, True)
        res = (self.sigmoid(self.a * (mean - self.b)) - self.sigmoid(-self.a * self.b)) / (
                self.sigmoid(self.a * (1 - self.b)) - self.sigmoid(-self.a * self.b))
        return res


class NN(torch.nn.Module):

    def __init__(self, n=2048, n_mid=1024,
                 n_out=1, dropout=0.2,
                 scoring=None,
                 ):
        super(NN, self).__init__()
        self.linear1 = torch.nn.Linear(n, n_mid)
        self.non_linearity = torch.nn.LeakyReLU()
        self.linear2 = torch.nn.Linear(n_mid, n_out)
        self.dropout = torch.nn.Dropout(dropout)
        if scoring:
            self.scoring = scoring
        else:
            self.scoring = torch.nn.Softmax() if n_out > 1 else torch.nn.Sigmoid()

    def forward(self, x):
        z = self.linear1(x)
        z = self.non_linearity(z)
        z = self.dropout(z)
        z = self.linear2(z)
        y_pred = self.scoring(z)
        return y_pred


class LogisticRegression(torch.nn.Module):
    def __init__(self, n=2048, n_out=1):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(n, n_out)
        self.scoring = torch.nn.Softmax() if n_out > 1 else torch.nn.Sigmoid()

    def forward(self, x):
        z = self.linear(x)
        y_pred = self.scoring(z)
        return y_pred


def regularization_loss(params,
                        reg_factor=0.005,
                        reg_alpha=0.5):
    params = [pp for pp in params if len(pp.shape) > 1]
    l1_reg = nn.L1Loss()
    l2_reg = nn.MSELoss()
    loss_reg = 0
    for pp in params:
        loss_reg += reg_factor * ((1 - reg_alpha) * l1_reg(pp, target=torch.zeros_like(pp)) + reg_alpha * l2_reg(pp, target=torch.zeros_like(pp)))
    return loss_reg


class MIL_NN(torch.nn.Module):

    def __init__(self, n=2048,
                 n_mid=1024,
                 n_classes=1,
                 dropout=0.1,
                 agg=None,
                 scoring=None,
                 ):
        super(MIL_NN, self).__init__()
        self.agg = agg if agg is not None else AttentionSoftMax(n)

        if n_mid == 0:
            self.bag_model = LogisticRegression(n, n_classes)
        else:
            self.bag_model = NN(n, n_mid, n_classes, dropout=dropout, scoring=scoring)

    def forward(self, bag_features, bag_lbls=None):
        print('')
        bag_feature, bag_att, bag_keys = list(zip(*[list(self.agg(ff.float())) + [idx] for idx, ff in enumerate(bag_features)]))
        # bag_feature, bag_att, bag_keys = list(zip(*[list(self.agg(ff.float())) + [idx] for idx, ff in (bag_features)]))
        bag_att = dict(zip(bag_keys, [a.detach().cpu() for a in bag_att]))
        bag_feature_stacked = torch.stack(bag_feature)
        y_pred = self.bag_model(bag_feature_stacked)
        return y_pred, bag_att, bag_keys


def calculate_metric(metric_fn, true_y, pred_y):
    # multi class problems need to have averaging method
    if "average" in inspect.getfullargspec(metric_fn).args:
        return metric_fn(true_y, pred_y, average="macro")
    else:
        return metric_fn(true_y, pred_y)


def print_scores(p, r, f1, a, batch_size):
    # just an utility printing function
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        print(f"\t{name.rjust(14, ' ')}: {sum(scores) / batch_size:.4f}")

start_ts = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

lr0 = 1e-4

# model:
model = MIL_NN().to(device)

# params you need to specify:
epochs = 200
train_loader, val_loader = get_data_loaders(padded_train, padded_test, 1, 1)
loss_function = torch.nn.BCELoss(reduction='mean') # your loss function, cross entropy works well for multi-class problems


#optimizer = optim.Adadelta(model.parameters())
optimizer = optim.SGD(model.parameters(), lr=lr0, momentum=0.9)

losses = []
batches = len(train_loader)
val_batches = len(val_loader)

# loop for every epoch (training + evaluation)
for epoch in range(epochs):
    total_loss = 0

    # progress bar (works in Jupyter notebook too!)
    progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batches)

    # ----------------- TRAINING  --------------------
    # set model to training
    model.train()
    for i, data in progress:
        X, y = data[0].to(device), data[1].to(device)
        # X = X.reshape([1, max_number_instance * 2048])
        y = y.type(torch.cuda.FloatTensor)
        # training step for single batch
        model.zero_grad()  # to make sure that all the grads are 0
        """
        model.zero_grad() and optimizer.zero_grad() are the same 
        IF all your model parameters are in that optimizer. 
        I found it is safer to call model.zero_grad() to make sure all grads are zero, 
        e.g. if you have two or more optimizers for one model.

        """
        outputs = model(X)  # forward
        loss = loss_function(outputs[0].squeeze(0), y)  # get loss
        loss.backward()  # accumulates the gradient (by addition) for each parameter.
        optimizer.step()  # performs a parameter update based on the current gradient

        # getting training quality data
        current_loss = loss.item()
        total_loss += current_loss

        # updating progress bar
        progress.set_description("Loss: {:.4f}".format(total_loss / (i + 1)))

    # releasing unceseccary memory in GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ----------------- VALIDATION  -----------------
    val_losses = 0
    precision, recall, f1, accuracy = [], [], [], []

    # set model to evaluating (testing)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            X, y = data[0].to(device), data[1].to(device)
            # X = X.reshape([1, max_number_instance * 2048])
            y = y.type(torch.cuda.FloatTensor)
            outputs = model(X)  # this get's the prediction from the network
            prediced_classes = outputs[0].detach().round()
            # y_pred.extend(prediced_classes.tolist())
            val_losses += loss_function(outputs[0].squeeze(0), y)

            # calculate P/R/F1/A metrics for batch
            for acc, metric in zip((precision, recall, f1, accuracy),
                                   (precision_score, recall_score, f1_score, accuracy_score)):
                acc.append(
                    calculate_metric(metric, y.cpu(), prediced_classes.cpu())
                )

    print(
        f"Epoch {epoch + 1}/{epochs}, training loss: {total_loss / batches}, validation loss: {val_losses / val_batches}")
    print_scores(precision, recall, f1, accuracy, val_batches)
    losses.append(total_loss / batches)  # for plotting learning curve
print(f"Training time: {time.time() - start_ts}s")






