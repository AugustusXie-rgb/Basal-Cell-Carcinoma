import numpy as np
import tensorflow as tf
from tensorflow import keras
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.preprocessing import image
import os

def get_img_array(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
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

    jet = cm.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = image.array_to_img(superimposed_img)

    superimposed_img.save(cam_path)

def Grad_CAM_UI(model_location,img_folder,output_folder,target_class_idx=0):
    last_conv_layer_name = "conv5_block3_out"
    print(model_location)
    model = ResNet101(
        weights=model_location,
        classes=2
    )
    model.layers[-1].activation = None
    img_list = os.listdir(img_folder)
    for i, img_name in enumerate(img_list):
        image_path = img_folder + img_name
        img_array = get_img_array(image_path)
        preds = model.predict(img_array)
        preds_out = preds[:, target_class_idx]
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=target_class_idx)
        output_path = output_folder + 'Grad_' + img_name
        save_and_display_gradcam(image_path, heatmap, cam_path=output_path)

# Grad_CAM_UI(model_location='/home/bfl/XieJun/keras_resnet/checkpoint/split_5/1/checkpoint-95e-val_accuracy_0.98.hdf5',
#             img_folder='./examine/selected/',
#             output_folder='./UI/')