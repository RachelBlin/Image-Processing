import os
import numpy as np
import imageio
import matplotlib.colors as mc
import colorsys



def rgb_to_hsv(path_rgb, path_hsv):
    """
    A function to convert an RGB image to an HSV image

    :param path_rgb: The path to the RGB image
    :param path_hsv: The path to the HSV image
    """
    image_rgb = imageio.imread(path_rgb)

    image_hsv = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 3))
    for m in range(image_rgb.shape[0]):
        for n in range(image_rgb.shape[1]):
            image_hsv[m, n, :] = colorsys.rgb_to_hsv(image_rgb[m,n,0], image_rgb[m,n,1], image_rgb[m,n,2])

    imageio.imwrite(path_hsv, image_hsv)

def hsv_to_rgb(path_hsv, path_rgb_verify):
    """
    A function to convert an HSV image to an RGB image

    :param path_hsv: The path to the HSV image
    :param path_rgb_verify: The path to the RGB image
    """
    image_hsv = imageio.imread(path_hsv)
    image_rgb_back = np.zeros((image_hsv.shape[0], image_hsv.shape[1], 3))
    for m in range(image_hsv.shape[0]):
        for n in range(image_hsv.shape[1]):
            image_rgb_back[m, n, :] = colorsys.hsv_to_rgb(image_hsv[m,n,0], image_hsv[m,n,1], image_hsv[m,n,2])

    imageio.imwrite(path_rgb_verify, image_rgb_back)

path_rgb = "/home/rblin/Documents/images_test/reflets/4.jpg"
path_rgb_verify = "/home/rblin/Documents/images_test/RGB.png"
path_hsv = "/home/rblin/Documents/images_test/HSV.png"