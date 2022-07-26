import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import pickle

train_X_BCC = os.listdir('/home/bfl/XieJun/keras_resnet/dataset/train/BCC_4_train/')
for i in range(len(train_X_BCC)):
    train_X_BCC[i] = '/home/bfl/XieJun/keras_resnet/dataset/train/BCC_4_train/' + train_X_BCC[i]
train_X_NS = os.listdir('/home/bfl/XieJun/keras_resnet/dataset/train/NS_4_train/')
for i in range(len(train_X_NS)):
    train_X_NS[i] = '/home/bfl/XieJun/keras_resnet/dataset/train/NS_4_train/' + train_X_NS[i]
train_X = train_X_BCC + train_X_NS
train_Y = np.empty((len(train_X), 2))
count = 0
for img_name in train_X_BCC:
    train_Y[count] = np.array((1, 0))
    count+=1
for img_name in train_X_NS:
    train_Y[count] = np.array((0, 1))
    count+=1

val_X_BCC = os.listdir('/home/bfl/XieJun/keras_resnet/dataset/val/BCC_4_val/')
for i in range(len(val_X_BCC)):
    val_X_BCC[i] = '/home/bfl/XieJun/keras_resnet/dataset/val/BCC_4_val/' + val_X_BCC[i]
val_X_NS = os.listdir('/home/bfl/XieJun/keras_resnet/dataset/val/NS_4_val/')
for i in range(len(val_X_NS)):
    val_X_NS[i] = '/home/bfl/XieJun/keras_resnet/dataset/val/NS_4_val/' + val_X_NS[i]
val_X = val_X_BCC + val_X_NS
val_Y = np.empty((len(val_X), 2))
count = 0
for img_name in val_X_BCC:
    val_Y[count] = np.array((1, 0))
    count+=1
for img_name in val_X_NS:
    val_Y[count] = np.array((0, 1))
    count+=1

def load_batch_image(img_path, train_set = True, target_size=(224, 224)):
    im = load_img(img_path, target_size=target_size)
    if train_set:
        return img_to_array(im) #converts image to numpy array
    else:
        return img_to_array(im)/255.0

def GET_DATASET_SHUFFLE(X_samples, Y_samples, batch_size, train_set=True):
    random.shuffle(X_samples)

    batch_num = int(len(X_samples) / batch_size)
    max_len = batch_num * batch_size
    X_samples = np.array(X_samples[:max_len])
    Y_samples = Y_samples[:max_len,:]
    print(X_samples.shape)

    X_batches = np.split(X_samples, batch_num)
    Y_batches = np.split(Y_samples, batch_num)

    for i in range(len(X_batches)):
        if train_set:
            x = np.array(list(map(load_batch_image, X_batches[i], [True for _ in range(batch_size)])))
        else:
            x = np.array(list(map(load_batch_image, X_batches[i], [False for _ in range(batch_size)])))
        # print(x.shape)
        y = np.array(Y_batches[i])
        yield x, y

model = ResNet101(
    weights=None,
    classes=2
)

filepath = '4-cut-checkpoint-{epoch:02d}e-val_accuracy_{val_accuracy:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, save_best_only=True, verbose=0, period=1, monitor='val_accuracy', mode='max')
callbacks_list = [checkpoint]

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9
)
opt=tf.keras.optimizers.Adam(learning_rate=lr_schedule)

batch_size = 16
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(GET_DATASET_SHUFFLE(train_X, train_Y, batch_size=batch_size, train_set=True), epochs=100, steps_per_epoch=len(train_X)//batch_size,
                    validation_data=GET_DATASET_SHUFFLE(val_X, val_Y, batch_size=batch_size, train_set=False), callbacks = callbacks_list)

model.save('4cut_model_1.h5')

with open('TrainHistory.txt','wb') as file_pi:
    pickle.dump(history.history, file_pi)