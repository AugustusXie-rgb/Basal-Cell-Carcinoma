import os, sys
import numpy as np
from scipy import ndimage
from PIL import Image
from tensorflow.keras.preprocessing import image
import random

def DataSet():
    train_path_BCC = './dataset/comp/4class_aug/1_cut/B/train/'
    train_path_NS = './dataset/comp/4class_aug/1_cut/N/train/'

    val_path_BCC = './dataset/comp/4class_aug/1_cut/B/val1/'
    val_path_NS = './dataset/comp/4class_aug/1_cut/N/val1/'

    imglist_train_BCC = os.listdir(train_path_BCC)
    imglist_train_NS = os.listdir(train_path_NS)
    imglist_val_BCC = os.listdir(val_path_BCC)
    imglist_val_NS = os.listdir(val_path_NS)

    X_train = np.empty((len(imglist_train_BCC) + len(imglist_train_NS), 224, 224, 3))
    Y_train = np.empty((len(imglist_train_BCC) + len(imglist_train_NS), 2))

    count = 0

    for img_name in imglist_train_BCC:
        img_path = train_path_BCC + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0

        X_train[count] = img
        Y_train[count] = np.array((1,0))
        count+=1

    for img_name in imglist_train_NS:
        img_path = train_path_NS + img_name
        img = image.load_img(img_path, target_size=(224,224))
        img = image.img_to_array(img) / 255.0

        X_train[count] = img
        Y_train[count] = np.array((0,1))
        count+=1

    X_val = np.empty((len(imglist_val_BCC) + len(imglist_val_NS), 224, 224, 3))
    Y_val = np.empty((len(imglist_val_BCC) + len(imglist_val_NS), 2))
    count = 0
    for img_name in imglist_val_BCC:
        img_path = val_path_BCC + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_val[count] = img
        Y_val[count] = np.array((1,0))
        count+=1
    for img_name in imglist_val_NS:
        img_path = val_path_NS + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_val[count] = img
        Y_val[count] = np.array((0,1))
        count+=1

    index = [i for i in range(len(X_train))]
    random.shuffle(index)
    X_train = X_train[index]
    Y_train = Y_train[index]

    index = [i for i in range(len(X_val))]
    random.shuffle(index)
    X_val = X_val[index]
    Y_val = Y_val[index]

    return X_train,Y_train,X_val,Y_val
