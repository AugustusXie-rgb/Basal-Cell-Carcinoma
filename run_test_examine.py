import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint
import csv
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

model = ResNet101(
    weights='/media/xiejun/data1/keras_resnet/checkpoint/NSC/group_11/16cut/checkpoint-1824e-val_loss_0.45e-val_acc_0.81.hdf5',
    classes=2
)

test_path = '/media/xiejun/data1/keras_resnet/dataset/comp/group_4/16cut_train/N/'
outputpath = '/media/xiejun/data1/keras_resnet/output/comp/group41_16cut_train_N .csv'
imglist_test = os.listdir(test_path)
imglist_test = sorted(imglist_test)
for img_name in imglist_test:
    img_path = test_path + img_name
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    np.argmax(model.predict(img))
    with open(outputpath,'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow([img_name, model.predict(img)])
        temp=model.predict(img)[0, 0]
    print(img_name)
print(outputpath)

# for i in NS_groups:
#     test_path = './dataset/NS_Groups/{}/'.format(i)
#     outputpath = './examine/NS_examine_{}.csv'.format(i)
#     imglist_test = os.listdir(test_path)
#     for img_name in imglist_test:
#         img_path = test_path + img_name
#         img = image.load_img(img_path, target_size=(224, 224))
#         img = image.img_to_array(img) / 255.0
#         img = np.expand_dims(img, axis=0)
#         np.argmax(model.predict(img))
#         with open(outputpath,'a+') as f:
#             csv_write = csv.writer(f)
#             csv_write.writerow([img_name, model.predict(img)])
