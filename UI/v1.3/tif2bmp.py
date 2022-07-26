import os,sys
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

def tif2bmp(file_dir,output_dir=None):
    imglist = os.listdir(file_dir)
    if not output_dir:
        output_dir = file_dir + 'bmp/'
    os.mkdir(output_dir)
    for imgname in imglist:
        img_path = file_dir + '/' + imgname
        img = image.load_img(img_path)
        img = image.img_to_array(img)
        img = img[:1000, :1000]
        image.save_img(output_dir + imgname, img, file_format='bmp', scale=True)

    return output_dir


# tif2bmp('/home/xiejun/keras_resnet/dataset/test/BCC_test/')
