from skimage import io, color
from sklearn.neighbors import NearestNeighbors
import numpy as np
import cv2
from scipy import ndimage
################################################################################################
#  Utlity functions for pre and post proccesing                                                #
#  This function were  inispired by https://github.com/foamliu/Colorful-Image-Colorization     #
################################################################################################


# Takes a rgb_image  and outputs a tuple of the corresponding input and ground truth for our model
def pre_proccess_image(rgb_image):

    h, w = rgb_image.shape[0] // 4, rgb_image.shape[1] // 4

    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)

    # The -128  on l a and b are to mean center the values
    l = lab_image[:, :, 0] - 128
    lab_image = cv2.resize(lab_image, (h, w), cv2.INTER_CUBIC)
    a = lab_image[:, :, 1].astype(np.int32)
    b = lab_image[:, :, 2].astype(np.int32)

    quantized_ab = np.load("resource/pts_in_hull.npy")

    a = a.ravel() - 128
    b = b.ravel() - 128
    ab = np.array([a, b]).T

    nbrs = NearestNeighbors(
        n_neighbors=5, algorithm='ball_tree').fit(quantized_ab)

    distances, indices = nbrs.kneighbors(ab)

    stdev = 5
    gaussed = np.exp(-distances ** 2 / (2 * stdev ** 2))
    gaussed = gaussed / np.sum(gaussed, axis=1)[..., np.newaxis]

    res = np.zeros((h*w, 313))
    idx_p = np.arange(h*w)[:, np.newaxis]

    res[idx_p, indices] = gaussed
    res = res.reshape((h, w, quantized_ab.shape[0]))

    return l, res


# Takes the input and predicted output of our model and converts it into the correspondnig RGB image.
# If the fixed_lightness parameter is set it wilil be used instead of the image input.
# This facilitates seeinig the model output
def post_proccess_image(l, data, fixed_lightness=None):
    h, w = data.shape[:2]
    temperature = 0.38
    data = data.reshape((h * w, 313))
    data = np.exp(np.log(data+1e-8) / temperature)
    data = data / np.sum(data, axis=1)[:, np.newaxis]

    # The +128  on l a and b are counteract the -128 in the preprocessing step
    l = l + 128
    quantized_ab = np.load("resource/pts_in_hull.npy")
    a = np.matmul(data, quantized_ab[:, 0]).reshape((h, w)) + 128
    b = np.matmul(data, quantized_ab[:, 1]).reshape((h, w)) + 128

    a = cv2.resize(a, l.shape, cv2.INTER_CUBIC)
    b = cv2.resize(b, l.shape, cv2.INTER_CUBIC)

    img = np.ones(l.shape + (3,), dtype=np.uint8)
    img[..., 0] = fixed_lightness or l
    img[..., 1] = a
    img[..., 2] = b

    rgb = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    return rgb


# Tests that postprocces(preprocces(a)) = a
if __name__ == "__main__":
    filename = "ele.JPEG"
    image = io.imread(filename)
    l, bop = pre_proccess_image(image)
    out = post_proccess_image(l, bop)
    io.imsave("outiimage.png", out)
