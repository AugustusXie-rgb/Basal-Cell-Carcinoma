import os, sys
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet101
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import pickle
from PIL import Image
import numpy
from matplotlib import cm
from sklearn import svm
from sklearn.preprocessing import minmax_scale
#import cv2
import math

def img_pro(im):
    im = im.resize((224, 224))
    im = np.asarray(im) / 255.0
    im = np.expand_dims(im, axis=0)
    return im

def run_model(model_path, img_path):

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    model_path_1cut = os.path.join( model_path, '1cut.hdf5')
    model_path_4cut = os.path.join( model_path, '4cut.hdf5')
    model_path_16cut = os.path.join( model_path, '16cut.hdf5')
    svm14_path = os.path.join( model_path, 'svm_model_14.pkl')
    svm416_path = os.path.join( model_path, 'svm_model_416.pkl')
    #print(model_path_1cut)
    model_1cut = ResNet101(weights=model_path_1cut, classes=2)
    model_4cut = ResNet101(weights=model_path_4cut, classes=2)
    model_16cut = ResNet101(weights=model_path_16cut, classes=2)
    with open(svm14_path, 'rb') as file:
        svm14 = pickle.load(file)
    with open(svm416_path, 'rb') as file:
        svm416 = pickle.load(file)

    imglist = os.listdir(img_path)
    imglist = sorted(imglist)
    BCC_seq = []

    hsv = cm.get_cmap('hsv', 256)
    hsv = hsv(np.linspace(0, 1, 256))
    hsv_bar = np.zeros((1, 172, 3))
    hsv_bar[:, :, :] = hsv[0:172, 0:3]
    hsv_bar = image.array_to_img(hsv_bar)
    hsv_bar = hsv_bar.resize((256, 1))
    hsv_bar = hsv_bar.transpose(method=Image.FLIP_LEFT_RIGHT)
    hsv = np.asarray(hsv_bar)
    for img_name in imglist:
        image_path = os.path.join(img_path,img_name)
        if os.path.isfile(image_path):
            img_ori = image.load_img(image_path, target_size=(1000, 1000))
            img_1 = img_ori.crop((0, 0, 500, 500))
            img_2 = img_ori.crop((0, 500, 500, 1000))
            img_3 = img_ori.crop((500, 0, 1000, 500))
            img_4 = img_ori.crop((500, 500, 1000, 1000))
            img_1_1 = img_1.crop((0, 0, 250, 250))
            img_1_2 = img_1.crop((0, 250, 250, 500))
            img_1_3 = img_1.crop((250, 0, 500, 250))
            img_1_4 = img_1.crop((250, 250, 500, 500))
            img_2_1 = img_2.crop((0, 0, 250, 250))
            img_2_2 = img_2.crop((0, 250, 250, 500))
            img_2_3 = img_2.crop((250, 0, 500, 250))
            img_2_4 = img_2.crop((250, 250, 500, 500))
            img_3_1 = img_3.crop((0, 0, 250, 250))
            img_3_2 = img_3.crop((0, 250, 250, 500))
            img_3_3 = img_3.crop((250, 0, 500, 250))
            img_3_4 = img_3.crop((250, 250, 500, 500))
            img_4_1 = img_4.crop((0, 0, 250, 250))
            img_4_2 = img_4.crop((0, 250, 250, 500))
            img_4_3 = img_4.crop((250, 0, 500, 250))
            img_4_4 = img_4.crop((250, 250, 500, 500))
            img_o = img_pro(img_ori)
            img_1 = img_pro(img_1)
            img_2 = img_pro(img_2)
            img_3 = img_pro(img_3)
            img_4 = img_pro(img_4)
            img_1_1 = img_pro(img_1_1)
            img_1_2 = img_pro(img_1_2)
            img_1_3 = img_pro(img_1_3)
            img_1_4 = img_pro(img_1_4)
            img_2_1 = img_pro(img_2_1)
            img_2_2 = img_pro(img_2_2)
            img_2_3 = img_pro(img_2_3)
            img_2_4 = img_pro(img_2_4)
            img_3_1 = img_pro(img_3_1)
            img_3_2 = img_pro(img_3_2)
            img_3_3 = img_pro(img_3_3)
            img_3_4 = img_pro(img_3_4)
            img_4_1 = img_pro(img_4_1)
            img_4_2 = img_pro(img_4_2)
            img_4_3 = img_pro(img_4_3)
            img_4_4 = img_pro(img_4_4)

            heat_16 = numpy.zeros((4, 4))
            heat_4 = numpy.zeros((2, 2))
            heat_16[0, 0] = model_16cut.predict(img_1_1)[0, 0]
            heat_16[0, 1] = model_16cut.predict(img_1_2)[0, 0]
            heat_16[1, 0] = model_16cut.predict(img_1_3)[0, 0]
            heat_16[1, 1] = model_16cut.predict(img_1_4)[0, 0]
            heat_16[0, 2] = model_16cut.predict(img_2_1)[0, 0]
            heat_16[0, 3] = model_16cut.predict(img_2_2)[0, 0]
            heat_16[1, 2] = model_16cut.predict(img_2_3)[0, 0]
            heat_16[1, 3] = model_16cut.predict(img_2_4)[0, 0]
            heat_16[2, 0] = model_16cut.predict(img_3_1)[0, 0]
            heat_16[2, 1] = model_16cut.predict(img_3_2)[0, 0]
            heat_16[3, 0] = model_16cut.predict(img_3_3)[0, 0]
            heat_16[3, 1] = model_16cut.predict(img_3_4)[0, 0]
            heat_16[2, 2] = model_16cut.predict(img_4_1)[0, 0]
            heat_16[2, 3] = model_16cut.predict(img_4_2)[0, 0]
            heat_16[3, 2] = model_16cut.predict(img_4_3)[0, 0]
            heat_16[3, 3] = model_16cut.predict(img_4_4)[0, 0]
            heat_4[0, 0] = model_4cut.predict(img_1)[0, 0]
            heat_4[0, 1] = model_4cut.predict(img_2)[0, 0]
            heat_4[1, 0] = model_4cut.predict(img_3)[0, 0]
            heat_4[1, 1] = model_4cut.predict(img_4)[0, 0]
            heat_1 = model_1cut.predict(img_o)[0, 0]

            input_x_416 = numpy.zeros((4, 6))
            input_x_416[0, 0] = heat_4[0, 0]
            input_x_416[0, 1] = heat_4[0, 0]
            input_x_416[0, 2] = heat_16[0, 0]
            input_x_416[0, 3] = heat_16[0, 1]
            input_x_416[0, 4] = heat_16[1, 0]
            input_x_416[0, 5] = heat_16[1, 1]
            input_x_416[1, 0] = heat_4[0, 1]
            input_x_416[1, 1] = heat_4[0, 1]
            input_x_416[1, 2] = heat_16[0, 2]
            input_x_416[1, 3] = heat_16[0, 3]
            input_x_416[1, 4] = heat_16[1, 2]
            input_x_416[1, 5] = heat_16[1, 3]
            input_x_416[2, 0] = heat_4[1, 0]
            input_x_416[2, 1] = heat_4[1, 0]
            input_x_416[2, 2] = heat_16[2, 0]
            input_x_416[2, 3] = heat_16[2, 1]
            input_x_416[2, 4] = heat_16[3, 0]
            input_x_416[2, 5] = heat_16[3, 1]
            input_x_416[3, 0] = heat_4[1, 1]
            input_x_416[3, 1] = heat_4[1, 1]
            input_x_416[3, 2] = heat_16[2, 2]
            input_x_416[3, 3] = heat_16[2, 3]
            input_x_416[3, 4] = heat_16[3, 2]
            input_x_416[3, 5] = heat_16[3, 3]
            output416 = svm416.predict_proba(input_x_416)[:, 1]

            input_x_14 = numpy.zeros((1, 6))
            input_x_14[0, 0] = heat_1
            input_x_14[0, 1] = heat_1
            input_x_14[0, 2:6] = output416[0:4]
            output14 = svm14.predict_proba(input_x_14)[:, 1]
            BCC_seq.append(output14[0])

            heatmap_dir = os.path.join(img_path,'heatmap')
            if not os.path.exists(heatmap_dir):
                os.mkdir(heatmap_dir)
            print(heatmap_dir)
            heat_16 = np.array([(x-np.min(heat_16))/(np.max(heat_16)-np.min(heat_16)) for x in heat_16])
            heat_4 = np.array([(x - np.min(heat_4)) / (np.max(heat_4) - np.min(heat_4)) for x in heat_4])
            heat_4 = Image.fromarray((heat_4*255).astype(np.uint8))
            heat_4 = heat_4.resize((4,4))
            heat_4 = np.asarray(heat_4).astype(float)
            # heat_4 = cv2.resize(heat_4, (4, 4), interpolation=cv2.INTER_CUBIC)
            heatmap = 255 * (0.25 * heat_4 + 0.75 * heat_16)
            heatmap = output14[0] * heatmap
            heatmap = Image.fromarray(heatmap.astype(np.uint8))
            heatmap = heatmap.resize((1000,1000))
            heatmap = np.asarray(heatmap).astype(np.int)
            #heatmap = np.round(cv2.resize(heatmap, (1000, 1000), interpolation=cv2.INTER_CUBIC))
            heatmap = np.clip(heatmap, 0, 255)
            heatmap = heatmap.astype(int)
            heat_mask = np.zeros((1000, 1000, 3))
            img_ori = np.asarray(img_ori).astype(np.float)
            hsv = hsv.astype(np.float)
            for i in range(1000):
                for j in range(1000):
                    heat_mask[i, j, :] = np.multiply(img_ori[i, j, :], hsv[0, heatmap[i, j], :])
            heat_mask = Image.fromarray(np.uint8(heat_mask/255))
            heat_mask_path = os.path.join(heatmap_dir,img_name)
            heat_mask.save(heat_mask_path)

    l = len(BCC_seq)
    state_4 = np.zeros(l)
    state_6 = np.zeros(l)
    state_8 = np.zeros(l)
    for i in range(l):
        if BCC_seq[i] >= 0.4:
            state_4[i] = 1
            if BCC_seq[i] >= 0.6:
                state_6[i] = 1
                if BCC_seq[i] >= 0.8:
                    state_8[i] = 1
    for i in range(1, l):
        if state_4[i] > 0:
            state_4[i] = state_4[i - 1] + 1
        if state_6[i] > 0:
            state_6[i] = state_6[i - 1] + 1
        if state_8[i] > 0:
            state_8[i] = state_8[i - 1] + 1
    for i in range(l - 1, 0, -1):
        if state_4[i] > 0 and state_4[i - 1] > 0:
            state_4[i - 1] = state_4[i]
        if state_6[i] > 0 and state_6[i - 1] > 0:
            state_6[i - 1] = state_6[i]
        if state_8[i] > 0 and state_8[i - 1] > 0:
            state_8[i - 1] = state_8[i]
    part_4 = np.clip(BCC_seq, 0.4, 0.6) - 0.4
    part_6 = np.clip(BCC_seq, 0.6, 0.8) - 0.6
    part_8 = np.clip(BCC_seq, 0.8, 1) - 0.8
    total_score = 0.001
    for i in range(l):
        total_score += part_4[i] * state_4[i] + part_6[i] * math.exp(state_6[i]) + part_8[i] * math.exp(2 * state_8[i])
    total_score = np.log(total_score)
    return total_score, BCC_seq, heatmap_dir
#
# a, b, c = run_model('model/', 'test_img/')
# print(a)
# print(b)
# print(c)