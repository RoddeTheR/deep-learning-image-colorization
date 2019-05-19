import h5py
import numpy as np
import tensorflow as tf
from skimage import io, color
import numpy as np
from imageproccesing import pre_proccess_image, post_proccess_image
from os import listdir
from sys import argv
import cv2

######################################################################
#  Creates a hdf5 file with the images in the given directory        #
######################################################################
mode = argv[1]  # "training"
image_dir = mode + "_images"

print("Counting valid images")
f_count = 0
for i, f in enumerate(listdir(image_dir)):
    if i % 1000 == 0:
        print("Processing: ", i)
    try:
        rgb = io.imread(image_dir + "/"+f)
        if rgb.ndim < 3:
            continue

        f_count += 1
        if f_count > 0:
            break
    except:
        print("Can't read it")

if f_count == 0:
    print("Found 0 valid files. Exiting")
    import sys
    sys.exit(0)

with h5py.File(mode + '.hdf5', 'w') as h5_f:
    dset = h5_f.create_dataset(
        mode + "_data", (f_count, 256, 256, 3), dtype='u1')
    all_of_the_lightness = []
    resses = []
    i = 0
    for f in listdir(image_dir):
        try:
            rgb = io.imread(image_dir + "/"+f)
            rgb = cv2.resize(rgb, (256, 256), cv2.INTER_CUBIC)
        except:
            print("Can't read it")
            continue
        if rgb.ndim < 3:
            continue

        dset[i] = rgb
        i += 1
        if i > 0:
            break
        if i % 1000 == 0:
            print(i)
