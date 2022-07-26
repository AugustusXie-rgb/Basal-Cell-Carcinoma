import os, sys
import numpy as np
import scipy
from scipy import ndimage
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image
import random
from DataSet import DataSet
import pickle
from tensorflow.keras.callbacks import ModelCheckpoint

#X_train,Y_train,X_val,Y_val = DataSet()
model = ResNet101(
    weights=None,
    classes=2
)


