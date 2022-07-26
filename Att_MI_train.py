import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
from matplotlib import pyplot as plt
import os
from PIL import Image


plt.style.use("ggplot")

def create_bags(B_bags_dir,N_bags_dir):
    B_Baglist = os.listdir(B_bags_dir)
    N_Baglist = os.listdir(N_bags_dir)
    Bags = []
    Bag_lbs=[]
    for B_bag_n in B_Baglist:
        print(B_bag_n)
        images = os.listdir(os.path.join(B_bags_dir,B_bag_n))
        I_bag = np.zeros((len(images),224,224))
        for idx , img in enumerate(images):
            I =Image.open(os.path.join(B_bags_dir,B_bag_n,img))
            I = I.resize((224,224))
            I_array = np.asarray(I)
            I.close()
            I_array = np.divide(I_array, 255.0)
            I_bag[idx,:,:] = I_array
        Bags.append(I_bag)
        Bag_lbs.append([1])

    for N_bag_n in N_Baglist:
        print(N_bag_n)
        images = os.listdir(os.path.join(N_bags_dir,N_bag_n))
        I_bag = np.zeros((len(images),224,224))
        for idx , img in enumerate(images):
            I =Image.open(os.path.join(N_bags_dir,N_bag_n,img))
            I = I.resize((224, 224))
            I_array = np.asarray(I)
            I.close()
            I_array = np.divide(I_array, 255.0)
            I_bag[idx,:,:] = I_array
        Bags.append(I_bag)
        Bag_lbs.append([0])
    return  (list(np.swapaxes(Bags, 0, 1)), np.array(Bag_lbs))

train_data, train_labels = create_bags(B_bags_dir='/home/bfl/XieJun/keras_resnet/dataset/NSC_bags/train/B/', N_bags_dir='/home/bfl/XieJun/keras_resnet/dataset/NSC_bags/train/N/')
val_data, val_labels = create_bags(B_bags_dir='/home/bfl/XieJun/keras_resnet/dataset/NSC_bags/val/B/', N_bags_dir='/home/bfl/XieJun/keras_resnet/dataset/NSC_bags/val/B/')

class MILAttentionLayer(layers.Layer):
    def __init__(
            self,
            weight_params_dim,
            kernel_initializer="glorot_uniform",
            kernel_regularizer=None,
            use_gated=False,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.weight_params_dim = weight_params_dim
        self.use_gated = use_gated

        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        self.v_init = self.kernel_initializer
        self.w_init = self.kernel_initializer
        self.u_init = self.kernel_initializer

        self.v_regularizer = self.kernel_regularizer
        self.w_regularizer = self.kernel_regularizer
        self.u_regularizer = self.kernel_regularizer

    def build(self, input_shape):
        input_dim = input_shape[0][1]

        self.v_weight_params = self.add_weight(
            shape=(input_dim, self.weight_params_dim),
            initializer=self.v_init,
            name="v",
            regularizer=self.v_regularizer,
            trainable=True,
        )

        self.w_weight_params = self.add_weight(
            shape=(self.weight_params_dim, 1),
            initializer=self.w_init,
            name="w",
            regularizer=self.w_regularizer,
            trainable=True,
        )

        if self.use_gated:
            self.u_weight_params = self.add_weight(
                shape=(input_dim, self.weight_params_dim),
                initializer=self.u_init,
                name="u",
                regularizer=self.u_regularizer,
                trainable=True,
            )
        else:
            self.u_weight_params = None

        self.input_built = True

    def call(self, inputs):
        instances = [self.compute_attention_scores(instance) for instance in inputs]
        alpha = tf.math.softmax(instances, axis=0)
        return [alpha[i] for i in range(alpha.shape[0])]

    def compute_attention_scores(self, instance):
        original_instance = instance
        instance = tf.math.tanh(tf.tensordot(instance, self.v_weight_params, axes=1))
        if self.use_gated:
            instance = instance * tf.math.sigmoid(
                tf.tensordot(original_instance, self.u_weight_params, axes=1)
            )
        return tf.tensordot(instance, self.w_weight_params, axes=1)

    def create_model(instance_shape):
        inputs, embeddings = [], []
        shared_dense_layer_1 = layers.Dense(128, activation="relu")
        shared_dense_layer_2 = layers.Dense(64, activation="relu")
        for _ in range(BAG_SIZE):
            inp = layers.Input(instance_shape)
            flatten = layers.Flatten()(inp)
            dense_1 = shared_dense_layer_1(flatten)
            dense_2 = shared_dense_layer_2(dense_1)
            inputs.append(inp)
            embeddings.append(dense_2)

        alpha = MILAttentionLayer(
            weight_params_dim=256,
            kernel_regularizer=keras.regularizers.l2(0.01),
            use_gated=True,
            name="alpha",
        )(embeddings)

        multiply_layers = [
            layers.multiply([alpha[i], embeddings[i]]) for i in range(len(alpha))
        ]

        concat = layers.concatenate(multiply_layers, axis=1)

        output = layers.Dense(2, activation="softmax")(concat)

        return keras.Model(inputs, output)

