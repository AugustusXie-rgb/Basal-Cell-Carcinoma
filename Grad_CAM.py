import numpy as np
import tensorflow as tf
from tensorflow import keras
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.preprocessing import image
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os

def get_img_array(img_path):
    img = image.load_img(image_path, target_size=(224, 224))
    array = image.img_to_array(img) / 255.0
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path='cam.jpg', alpha=0.4):
    img = image.load_img(img_path)
    img = image.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)
    heatmap = tf.expand_dims(heatmap, axis=2)
    heatmap = np.concatenate((heatmap, heatmap, heatmap), axis=2)

    # jet = cm.get_cmap("jet")
    #
    # jet_colors = jet(np.arange(256))[:, :3]
    # jet_heatmap = jet_colors[heatmap]
    #
    # jet_heatmap = image.array_to_img(jet_heatmap)
    # jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    # jet_heatmap = image.img_to_array(jet_heatmap)

    # superimposed_img = jet_heatmap * alpha + img
    # superimposed_img = image.array_to_img(superimposed_img)

    heatmap = image.array_to_img(heatmap)
    heatmap.save(cam_path)
    # plt.imshow(superimposed_img)
    # plt.show()

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

target_class_idx = 0
last_conv_layer_name = "conv5_block3_out"
model = ResNet101(
    weights='/media/xiejun/data1/keras_resnet/checkpoint/NSC/group_52/1cut/checkpoint-750e-val_loss_0.06e-val_acc_0.98.hdf5',
    classes=2
)
model.layers[-1].activation = None

image_folder = '/media/xiejun/data1/keras_resnet/Grad_CAM/Original/'
outputfolder = '/media/xiejun/data1/keras_resnet/Grad_CAM/heatmap/'
img_list = os.listdir(image_folder)
for i,img_name in enumerate(img_list):
    image_path = image_folder + img_name
    img_array = get_img_array(image_path)

    preds = model.predict(img_array)
    preds_out = preds[:, target_class_idx]
    # print("Predicted:", preds_out)

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=target_class_idx)

# plt.matshow(heatmap)
# plt.show()
    output_path = outputfolder + 'Grad_' + img_name
    save_and_display_gradcam(image_path, heatmap, cam_path=output_path)