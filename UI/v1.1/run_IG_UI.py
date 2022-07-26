import matplotlib.pylab as plt
import os
import numpy as np
import tensorflow as tf
import matplotlib.cm as cm
import tensorflow_hub as hub
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.preprocessing import image

def read_image(file_name):
    image = tf.io.read_file(file_name)
    image = tf.image.decode_bmp(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_pad(image, target_height=224, target_width=224)
    return image

def interpolate_images(baseline, image, alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(image, axis=0)
    delta = input_x - baseline_x
    images = baseline_x + alphas_x * delta
    return images

def compute_gradients(images, target_class_idx,model):
    with tf.GradientTape() as tape:
        tape.watch(images)
        #print(images.shape)
        probs = model(images)#[:, target_class_idx]
        probs = probs[:, target_class_idx]
    return tape.gradient(probs, images)

def integral_approximation(gradients):
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    ig = tf.math.reduce_mean(grads, axis=0)
    return ig


def integrated_gradients(baseline,
                         image,
                         target_class_idx,
                         model,
                         m_steps=50,
                         batch_size=50):
    #print(image.shape)
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)
    gradient_batches = tf.TensorArray(tf.float32, size=m_steps+1)

    for alpha in tf.range(0, len(alphas), batch_size):
        from_ = alpha
        to = tf.minimum(from_ + batch_size, len(alphas))
        alpha_batch = alphas[from_:to]

        interpolated_path_input_batch = interpolate_images(baseline=baseline, image=image, alphas=alpha_batch)
        gradient_batch = compute_gradients(images=interpolated_path_input_batch, target_class_idx=target_class_idx, model=model)
        gradient_batches = gradient_batches.scatter(tf.range(from_, to), gradient_batch)

    total_gradients = gradient_batches.stack()
    avg_gradients = integral_approximation(gradients=total_gradients)
    ig_attributions = (image - baseline) * avg_gradients

    return ig_attributions


def plot_img_attributions(attributions,
                          target_class_idx,
                          img_path,
                          img_name,
                          outputfolder,
                          m_steps=50,
                          cmap=None,
                          overlay_alpha=0.4):
    attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)
    img_ori = image.load_img(img_path)
    img_ori = image.img_to_array(img_ori)
    max_attribution_mask = np.max(attribution_mask)
    attribution_mask = 1 * attribution_mask / max_attribution_mask
    attribution_mask = np.uint8(255*attribution_mask)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    attribution_mask = jet_colors[attribution_mask]
    attribution_mask = image.array_to_img(attribution_mask)
    attribution_mask = attribution_mask.resize((img_ori.shape[1], img_ori.shape[0]))
    attribution_mask = image.img_to_array(attribution_mask)
    superimposed_img = attribution_mask * overlay_alpha + img_ori
    superimposed_img = image.array_to_img(superimposed_img)

    outputpath = outputfolder + 'IG_' + img_name.replace('bmp','png')
    superimposed_img.save(outputpath)

def run_IG_UI(model_location,img_folder,output_folder,target_class_idx=0):
    img_list = os.listdir(img_folder)
    ig_score = []
    ig_sum = tf.zeros(shape=(224, 224, 3))
    model = ResNet101(
        weights=model_location,
        classes=2
    )
    for i, img_name in enumerate(img_list):
        count = 0;
        # print(i)
        for j in range(50):
            baseline = tf.random.uniform(shape=(224, 224, 3), minval=0, maxval=None, dtype=tf.dtypes.float32)
            # print(i)
            image_path = img_folder + img_name
            img = image.load_img(image_path, target_size=(224, 224))
            img = image.img_to_array(img) / 255.0
            img_temp = np.expand_dims(img, axis=0)
            baseline_temp = image.img_to_array(baseline) / 255.0
            baseline_temp = np.expand_dims(baseline, axis=0)
            img = tf.convert_to_tensor(img, dtype=tf.float32)

            probs_img = model.predict(img_temp)
            probs_img = probs_img[:, target_class_idx]
            probs_baseline = model.predict(baseline_temp)
            probs_baseline = probs_baseline[:, target_class_idx]

            ig_attributions = integrated_gradients(baseline=baseline,
                                                   image=img,
                                                   target_class_idx=target_class_idx,
                                                   model=model,
                                                   m_steps=240,
                                                   batch_size=1)
            if probs_baseline <= 0.1:
                ig_sum += ig_attributions
                count += 1
            else:
                print(probs_baseline, ' ', j)

        ig_sum = ig_sum / count

        plot_img_attributions(attributions=ig_sum,
                              target_class_idx=target_class_idx,
                              img_path=image_path,
                              outputfolder=output_folder,
                              m_steps=240,
                              cmap=plt.cm.jet,
                              overlay_alpha=0.4,
                              img_name=img_name)

# run_IG_UI(model_location='./checkpoint/split_5/1/checkpoint-95e-val_accuracy_0.98.hdf5',
#           img_folder='./examine/selected/',
#           output_folder='./UI/')


