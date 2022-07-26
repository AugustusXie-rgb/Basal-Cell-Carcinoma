import os, sys
import numpy as np
import scipy
from scipy import ndimage
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image
import random
from DataSet import DataSet
import pickle
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.preprocessing.image import ImageDataGenerator

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

datagen = ImageDataGenerator(rescale=1./255)
train_it = datagen.flow_from_directory('/media/xiejun/data1/keras_resnet/dataset/comp/group_5/1cut_train/', class_mode='categorical', batch_size=16)
val_it = datagen.flow_from_directory('/media/xiejun/data1/keras_resnet/dataset/comp/group_5/1cut_val2/', class_mode='categorical', batch_size=16)
# test_it = datagen.flow_from_directory('/media/xiejun/data1/keras_resnet/dataset/comp_data/4class/Aug/16_cut/val2/', class_mode='categorical', batch_size=16)

# X_train,Y_train,X_val,Y_val = DataSet()
model = ResNet101(
    weights=None,
    classes=2
)

filepath = '/media/xiejun/data1/keras_resnet/checkpoint/comp/group_52/1cut/checkpoint-{epoch:02d}e-val_loss_{val_loss:.2f}e-val_acc_{val_accuracy:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, save_best_only=True, verbose=0, period=1, monitor='val_loss', mode='min')
callbacks_list = [checkpoint]

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.9
)
opt=tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

for r in range(20):
    history = model.fit_generator(train_it, steps_per_epoch=16, epochs=2000, validation_data=(val_it), validation_steps=8, callbacks=callbacks_list)

    # model.save('/media/xiejun/data1/keras_resnet/checkpoint/NSC/group_1/1cut/model.h5')
    history_path = '/media/xiejun/data1/keras_resnet/trainhistory/comp_group52_1cut_' + str(r) + '.txt'
    with open(history_path,'wb') as file_pi:
        pickle.dump(history.history, file_pi)
