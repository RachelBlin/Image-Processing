import os
import imageio
import numpy as np
import cv2

def get_saliency_map(image):
    """

    :param path_folder:
    :return:
    """

    saliency_map = image

    for i in range(saliency_map.shape[0]):
        for j in range(saliency_map.shape[1]):
            saliency_map[i, j] = compute_pixel_distance(image[i, j], image)

    return saliency_map

def compute_pixel_distance(pix, image):

    dist = np.absolute(pix - image)
    dist = np.sum(dist)

    return dist

def main(path_folder, path_process):

    images = sorted(os.listdir(path_folder))

    for im in images:
        image = imageio.imread(os.path.join(path_folder, im))
        saliency_map = get_saliency_map(image)
        imageio.imwrite(os.path.join(path_process, im), saliency_map)


path_folder = "/home/rblin/Documents/Databases/saliency_test/images_test"
path_process = "/home/rblin/Documents/Databases/saliency_test/saliency_maps"

main(path_folder, path_process)
