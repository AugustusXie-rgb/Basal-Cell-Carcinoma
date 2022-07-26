import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet101
from DataGenerator import DataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle

params = {'dim': (224, 224), 'batch_size': 64, 'n_classes': 2, 'n_channels': 3, 'shuffle': True}

train_BCC_path = '/home/bfl/XieJun/keras_resnet/dataset/train/BCC_4_train/'
train_NS_path = '/home/bfl/XieJun/keras_resnet/dataset/train/NS_4_train/'
val_BCC_path = '/home/bfl/XieJun/keras_resnet/dataset/val/BCC_4_val/'
val_NS_path = '/home/bfl/XieJun/keras_resnet/dataset/val/NS_4_val/'

train_X_BCC = os.listdir(train_BCC_path)
for i in range(len(train_X_BCC)):
    train_X_BCC[i] = train_BCC_path + train_X_BCC[i]
train_X_NS = os.listdir(train_NS_path)
for i in range(len(train_X_NS)):
    train_X_NS[i] = train_NS_path + train_X_NS[i]
val_X_BCC = os.listdir(val_BCC_path)
for i in range(len(val_X_BCC)):
    val_X_BCC[i] = val_BCC_path + val_X_BCC[i]
val_X_NS = os.listdir(val_NS_path)
for i in range(len(val_X_NS)):
    val_X_NS[i] = val_NS_path + val_X_NS[i]

partition = {'train': train_X_BCC + train_X_NS, 'validation': val_X_BCC + val_X_NS}
labels = {}
for i in train_X_BCC:
    labels[i] = 1
for i in train_X_NS:
    labels[i] = 0
for i in val_X_BCC:
    labels[i] = 1
for i in val_X_NS:
    labels[i] = 0

train_generator = DataGenerator(partition['train'], labels, **params)
val_generator = DataGenerator(partition['validation'], labels, **params)

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

# batch_size = 16
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator, epochs=100,
                    validation_data=val_generator, callbacks = callbacks_list)

model.save('4cut_model_1.h5')

with open('TrainHistory.txt','wb') as file_pi:
    pickle.dump(history.history, file_pi)