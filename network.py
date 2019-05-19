import tensorflow as tf
from skimage import io, color
import numpy as np
from imageproccesing import pre_proccess_image, post_proccess_image
from time import time
import random
import h5py

################################
# Creates and trains the model #
################################
log_dir = "logs"
period_save_model = 200
model_save_folder = "models"

initializer = "he_normal"
regularizer = tf.keras.regularizers.l2(1e-4)


# Returns a Conv2D layer that defaults to the most common parameters
def conv2Dlayer(output_size, kernel_shape, strides=(1, 1), dilation_rate=(1, 1), activation=tf.nn.relu):
    return tf.keras.layers.Conv2D(output_size, kernel_shape, strides=strides, padding="same", kernel_initializer=initializer,
                                  kernel_regularizer=regularizer, activation=activation, dilation_rate=dilation_rate)


with tf.device("/gpu:0"):
    model = tf.keras.models.Sequential([
        # Block 1
        tf.keras.layers.Conv2D(64, (3, 3), padding="same", kernel_initializer=initializer,
                               kernel_regularizer=regularizer,
                               activation=tf.nn.relu, input_shape=(None, None, 1)),
        conv2Dlayer(64, (3, 3), strides=(2, 2)),
        tf.keras.layers.BatchNormalization(),

        # Block 2
        conv2Dlayer(128, (3, 3)),
        conv2Dlayer(128, (3, 3), strides=(2, 2)),
        tf.keras.layers.BatchNormalization(),

        # Block 3
        conv2Dlayer(256, (3, 3)),
        conv2Dlayer(256, (3, 3)),
        conv2Dlayer(256, (3, 3), strides=(2, 2)),
        tf.keras.layers.BatchNormalization(),

        # Block 4
        conv2Dlayer(512, (3, 3)),
        conv2Dlayer(512, (3, 3)),
        conv2Dlayer(512, (3, 3)),
        tf.keras.layers.BatchNormalization(),

        # Block 5
        conv2Dlayer(512, (3, 3), dilation_rate=(2, 2)),
        conv2Dlayer(512, (3, 3), dilation_rate=(2, 2)),
        conv2Dlayer(512, (3, 3), dilation_rate=(2, 2)),
        tf.keras.layers.BatchNormalization(),

        # Block 6
        conv2Dlayer(512, (3, 3), dilation_rate=(2, 2)),
        conv2Dlayer(512, (3, 3), dilation_rate=(2, 2)),
        conv2Dlayer(512, (3, 3), dilation_rate=(2, 2)),
        tf.keras.layers.BatchNormalization(),

        # Block 7
        conv2Dlayer(256, (3, 3)),
        conv2Dlayer(256, (3, 3)),
        conv2Dlayer(256, (3, 3)),
        tf.keras.layers.BatchNormalization(),

        # Block 8
        tf.keras.layers.UpSampling2D(size=(2, 2)),
        conv2Dlayer(128, (3, 3)),
        conv2Dlayer(128, (3, 3)),
        conv2Dlayer(128, (3, 3)),

        # -------
        conv2Dlayer(313, (1, 1), activation="softmax")

    ])

# Generates data for the model from a given .hdf5 file
# Batch size and shuffle can be set.
# The training images are cropped to a random 176x176 square


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, d_set, batch_size=1, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.training_mode = d_set == "training"

        self.f = h5py.File(d_set + '.hdf5', 'r')
        self.data_set = self.f[d_set + '_data']

        self.set_len = len(self.data_set)
        self.index_order = np.random.permutation(
            int(np.floor(self.set_len / self.batch_size)))
        self.orig_h = self.data_set.shape[1]
        self.orig_w = self.data_set.shape[2]
        if self.training_mode:
            self.training_w = 176
            self.training_h = 176
        else:
            self.training_w = self.orig_w
            self.training_h = self.orig_h
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.set_len / self.batch_size))

    def __getitem__(self, index):
        i = self.index_order[index // self.batch_size] * self.batch_size
        ret = self.data_set[i: i + self.batch_size]
        x_res = np.empty([self.batch_size, self.training_h,
                          self.training_w], dtype=np.uint8)
        y_res = np.empty([self.batch_size, self.training_h //
                          4, self.training_w // 4, 313], dtype=np.float64)
        for j in range(self.batch_size):
            h_start = np.random.randint(self.orig_h - self.training_h + 1)
            w_start = np.random.randint(self.orig_w - self.training_w + 1)

            x_res[j], y_res[j] = pre_proccess_image(ret[j, h_start: h_start + self.training_h,
                                                        w_start: w_start + self.training_w, :])
        return x_res[:, :, :, np.newaxis], y_res

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.index_order)


model.summary()

opt = tf.keras.optimizers.Adam(
    lr=1e-4, beta_1=0.9, beta_2=0.99, decay=1e-3)

model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir="{}/{}".format(log_dir, time()))
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    model_save_folder+"/"+"model{epoch:08d}.h5", period=period_save_model)
lr_decay = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=50, verbose=1)

tg = DataGenerator("training")
vg = DataGenerator("validation")
model.fit_generator(tg, validation_data=vg, epochs=300,
                    callbacks=[tensorboard, model_checkpoint, lr_decay],
                    workers=8, use_multiprocessing=False, max_queue_size=32)

model.save('model.h5')
