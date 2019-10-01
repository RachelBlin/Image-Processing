import os
import imageio
import numpy as np
from shutil import copyfile

def compute_noise(path_folder, path_traitement):
    """
    ToDo
    :param path_folder:
    :param path_traitement:
    :return:
    """
    imgs_polar = sorted(os.listdir(path_folder))
    for k in range(len(imgs_polar)):
        image = imageio.imread(path_folder + "/" + imgs_polar[k])

        I = np.zeros((4, int(image.shape[0]/2), int(image.shape[1]/2)))
        # On va donc retrouver P0, P45, P90, P135

        I[0] = image[0:image.shape[0]:2,1:image.shape[0]:2] / np.max(image[0:image.shape[0]:2,1:image.shape[0]:2]) * 255 # I0
        I[1] = image[0:image.shape[0]:2,0:image.shape[0]:2] / np.max(image[0:image.shape[0]:2,0:image.shape[0]:2]) * 255 # I45
        I[2] = image[1:image.shape[0]:2,0:image.shape[0]:2] / np.max(image[1:image.shape[0]:2,0:image.shape[0]:2]) * 255 # I90
        I[3] = image[1:image.shape[0]:2,1:image.shape[0]:2] / np.max(image[1:image.shape[0]:2,1:image.shape[0]:2]) * 255 # I135

        A = np.array([[1,1,0],[1,0,1],[1,-1,0],[1,0,-1]])