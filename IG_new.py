import numpy as np
from xml.dom import minidom
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications.resnet import preprocess_input


def get_img_array(img_path, size=(224, 224)):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def get_gradients(img_input, model, top_pred_idx):
    """Computes the gradients of outputs w.r.t input image.

    Args:
        img_input: 4D image tensor
        top_pred_idx: Predicted label for the input image

    Returns:
        Gradients of the predictions w.r.t img_input
    """
    images = tf.cast(img_input, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(images)
        preds = model(images)
        top_class = preds[:, top_pred_idx]

    grads = tape.gradient(top_class, images)
    return grads


def get_integrated_gradients(img_input,model, top_pred_idx, baseline=None, num_steps=50, model_n = 'VGG16'):
    """Computes Integrated Gradients for a predicted label.

    Args:
        img_input (ndarray): Original image
        top_pred_idx: Predicted label for the input image
        baseline (ndarray): The baseline image to start with for interpolation
        num_steps: Number of interpolation steps between the baseline
            and the input used in the computation of integrated gradients. These
            steps along determine the integral approximation error. By default,
            num_steps is set to 50.

    Returns:
        Integrated gradients w.r.t input image
    """
    # If baseline is not provided, start with a black image
    # having same size as the input image.
    if baseline is None:
        baseline = np.zeros(img_size).astype(np.float32)
    else:
        baseline = baseline.astype(np.float32)

    # 1. Do interpolation.
    img_input = img_input.astype(np.float32)

    #img_input = preprocess_input(img_input)
    #baseline = preprocess_input(baseline)


    interpolated_image = [
        baseline + (step / num_steps) * (img_input - baseline)
        for step in range(num_steps + 1)
    ]
    interpolated_image = np.array(interpolated_image).astype(np.float32)

    # 2. Preprocess the interpolated images

   # exec('interpolated_image = {}_preprocess_input(interpolated_image)'.format(model_n))

    # 3. Get the gradientsexec('preprocess_input = {}_preprocess_input'.format(model_n))

    if num_steps>50:
        grads = np.zeros(interpolated_image.shape)
        iter_num=num_steps//50
        for iter in range(iter_num):
            grads[iter*50:(iter+1)*50] = get_gradients(interpolated_image[iter*50:(iter+1)*50], model, top_pred_idx=top_pred_idx)
        grads[iter_num*50:]=get_gradients(interpolated_image[iter_num*50:], model, top_pred_idx=top_pred_idx)
    else:
        grads = get_gradients(interpolated_image, model, top_pred_idx=top_pred_idx)
    # 4. Approximate the integral using the trapezoidal rule
    #grads = grads.numpy()
    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = np.mean(grads, axis=0)

    # 5. Calculate integrated gradients and return
    integrated_grads = ( img_input- baseline) * avg_grads
    return integrated_grads

def plot_img_attributions(baseline,
                          image,
                          target_class_idx,
                          m_steps=50,
                          cmap=None,
                          overlay_alpha=0.4):
    attributions = integrated_gradients(baseline=baseline, image=image, target_class_idx=target_class_idx, m_steps=m_steps)
    attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)
    fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(8, 8))

    axs[0, 0].set_title('Baseline image')
    axs[0, 0].imshow(baseline)
    axs[0, 0].axis('off')

    axs[0, 1].set_title('Original image')
    axs[0, 1].imshow(image)
    axs[0, 1].axis('off')

    axs[1, 0].set_title('Attribution mask')
    axs[1, 0].imshow(attribution_mask, cmap=cmap)
    axs[1, 0].axis('off')

    axs[1, 1].set_title('Overlay')
    axs[1, 1].imshow(attribution_mask, cmap=cmap)
    axs[1, 1].imshow(image, alpha=overlay_alpha)
    axs[1, 1].axis('off')

    plt.tight_layout()
    return fig

target_class_idx = 0
model = ResNet101(
    weights='./checkpoint/split_5/2/checkpoint-75e-val_accuracy_0.97.hdf5',
    classes=2
)

# Size of the input image
img_size = (224, 224, 3)

img_path = './examine/BCCexamine104/v0000003 (105).bmp'

steps = 50

imfile = os.path.join(img_path)
img = get_img_array(imfile)
# 2. Keep a copy of the original image
orig_img = np.copy(img[0]).astype(np.uint8)

igrads = get_integrated_gradients(
    img_input=np.copy(orig_img),
    model = model,
    top_pred_idx = target_class_idx,
    baseline=None,
    num_steps=steps)
print(igrads)

_ = plot_img_attributions(image=np.copy(orig_img),
                          baseline=None,
                          target_class_idx=target_class_idx,
                          m_steps=)

