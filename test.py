import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.callbacks import ModelCheckpoint

model = ResNet101(weights='imagenet')
