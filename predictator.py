import tensorflow as tf
from skimage import io, color
import numpy as np
from imageproccesing import pre_proccess_image, post_proccess_image
from os import listdir
from sys import argv

#################################################################################
#  Predicts all images located in the "to_predict folder" using the "model.h5"  #
#################################################################################

model_file = argv[1]  # "model.h5"
predict_folder = "to_predict"

model = tf.keras.models.load_model(model_file)


# Reads the images and saves the L layers
l_layers = []
for f in listdir(predict_folder):
    rgb = io.imread(f"{predict_folder}/"+f)
    if rgb.ndim < 3:
        continue
    l, _ = pre_proccess_image(rgb)
    l_layers.append(l)


l_layers = np.array(l_layers)
x_vals = l_layers[:, :, :, np.newaxis]

y_vals = model.predict(x_vals)


# Saves the combination of the original L layer and the predicted A-B layers
# Also saves an image of L=MAX and A-B layers
for i, y in enumerate(y_vals):

    img = post_proccess_image(l_layers[i], y)
    io.imsave(f"{predict_folder}/img{i}.png", img)
    ab = post_proccess_image(l_layers[i], y, fixed_lightness=255)
    io.imsave(f"{predict_folder}/img_map{i}.png", ab)
