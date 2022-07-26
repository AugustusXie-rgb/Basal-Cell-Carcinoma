import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.callbacks import ModelCheckpoint
import csv

model = ResNet101(
    weights='./checkpoint/split_5/2/checkpoint-75e-val_accuracy_0.97.hdf5',
    classes=2
)
#model.load_weights('checkpoint-81e-val_accuracy_0.98.hdf5')

test_path_BCC = './dataset/split/split_5/BCC_val1/'
test_path_NS = './dataset/split/split_5/NS_val1/'
outputpath_BCC = 'split5_val1_BCC.csv'
outputpath_NS = 'split5_val1_NS.csv'

imglist_test_BCC = os.listdir(test_path_BCC)
imglist_test_NS = os.listdir(test_path_NS)

for img_name in imglist_test_BCC:
    img_path = test_path_BCC + img_name
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    #temp = model.predict(img)
    #print(model.predict(img))
    np.argmax(model.predict(img))
    with open(outputpath_BCC,'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow([img_name, model.predict(img)])

for img_name in imglist_test_NS:
    img_path = test_path_NS + img_name
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    #print(model.predict(img))
    with open(outputpath_NS, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow([img_name, model.predict(img)])