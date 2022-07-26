import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.callbacks import ModelCheckpoint
import csv

model = ResNet101(
    weights='./checkpoint/split_5/1/checkpoint-95e-val_accuracy_0.98.hdf5',
    classes=2
)

# BCC_groups = [120, 79, 101, 57, 5, 110, 98, 26, 59, 122, 104, 100, 94]
# NS_groups = [246, 158, 116, 255, 29, 223, 265, 162, 183, 177, 231, 240, 85, 219, 253, 197, 192, 13, 4, 117, 166, 98, 37, 18, 61, 100, 84, 81, 35]

# for i in BCC_groups:
    # test_path = './dataset/BCC_examine/{}/'.format(i)
    # outputpath = './examine/BCC_examine_{}.csv'.format(i)
test_path = './dataset/BCC_group_crop/40/'
outputpath = './examine/BCC_crop_examine40.csv'
imglist_test = os.listdir(test_path)
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