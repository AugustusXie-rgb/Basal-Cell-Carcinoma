import os,sys
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import math

def quality_indicator(input_image_path):
    input_image = image.load_img(input_image_path)
    mean_log_comb = 10.6141
    std_log_comb = 0.8647
    threshold_1 = mean_log_comb - 3 * std_log_comb
    threshold_2 = mean_log_comb - 2 * std_log_comb
    threshold_3 = mean_log_comb - std_log_comb
    img = image.img_to_array(input_image)
    img = tf.math.reduce_mean(img, axis=2)
    background = np.zeros((1000,1000))
    mse = np.mean((img-background)**2)
    psnr = 20*math.log10(1/math.sqrt(mse))
    # print(psnr)
    mean_img = np.mean(img)
    std_img = np.std(img)
    comb = np.log(mean_img * std_img * (-1) * psnr)
    # print(comb)
    indicator = None
    if comb < threshold_1:
        indicator = 'red'
    elif comb < threshold_2:
        indicator = 'orange'
    elif comb < threshold_3:
        indicator = 'yellow'
    else:
        indicator = 'green'
    return indicator

# img = image.load_img('/home/xiejun/keras_resnet/dataset/test/BCC_test/bmp/v0000000 (33).bmp')
# print(quality_indicator(input_image=img))