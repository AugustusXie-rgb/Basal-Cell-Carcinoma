import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.callbacks import ModelCheckpoint

def run_test_UI(model_location, img_folder):
    model = ResNet101(
        weights = model_location,
        classes = 2
    )
    imglist_test = os.listdir(img_folder)
    output_list = []
    for img_name in imglist_test:
        img_path = img_folder + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        np.argmax(model.predict(img))
        temp = model.predict(img)[0, 0]
        output_list.append(temp)

    avg_score = np.mean(output_list)
    return output_list,avg_score

# a, b = run_test_UI(model_location='./checkpoint/split_5/1/checkpoint-95e-val_accuracy_0.98.hdf5',img_folder='./dataset/BCC_group_crop/40/')
# print(a)
# print(b)