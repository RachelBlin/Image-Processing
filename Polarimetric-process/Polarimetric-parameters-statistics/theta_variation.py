import numpy as np
import imageio
import matplotlib.pyplot as plt
import os

def get_intensities(imgs_polar, k, path_process):
    """
    Split a Raw polarimetric images into the four intensities

    :param image: The image to be processed
    :param imgs_polar: The list containing all the polarimetric images in the folder
    :param k: The index of the image in the imgs_polar list
    :return: A vector containing all the intensities
    """

    image = imageio.imread(path_folder + "/" + imgs_polar[k])

    I = np.zeros((4, int(image.shape[0] / 2), int(image.shape[1] / 2)))

    I[0] = image[0:image.shape[0]:2, 1:image.shape[0]:2] / np.max(
        image[0:image.shape[0]:2, 1:image.shape[0]:2]) * 255  # I0
    I[1] = image[0:image.shape[0]:2, 0:image.shape[0]:2] / np.max(
        image[0:image.shape[0]:2, 0:image.shape[0]:2]) * 255  # I45
    I[2] = image[1:image.shape[0]:2, 0:image.shape[0]:2] / np.max(
        image[1:image.shape[0]:2, 0:image.shape[0]:2]) * 255  # I90
    I[3] = image[1:image.shape[0]:2, 1:image.shape[0]:2] / np.max(
        image[1:image.shape[0]:2, 1:image.shape[0]:2]) * 255  # I135

    return I, image

def get_stokes_parameters(I, imgs_polar, k, path_process):
    """
    Compute the Stokes parameters

    :param I: The intensities of the image
    :param imgs_polar: The list containing all the polarimetric images in the folder
    :param k: The index of the image in the imgs_polar list
    :return: A vector containing the Stokes parameters
    """

    Stokes = np.zeros((3, I[0].shape[0], I[0].shape[1]))

    Stokes[0] = I[0] + I[2]
    Stokes[1] = I[0] - I[2]
    Stokes[2] = I[1] - I[3]

    return Stokes

def computing_I_AS(I, Stokes, A):
    # Verifying the polarimetric constraint I = AS by plotting an histogram of ||I-AS||/(||I|| + ||AS||) for each
    # pixel of the image
    AS = np.zeros((4,500,500))
    for i in range(500):
        for j in range(500):
            AS[:,i,j] = A.dot(Stokes[:,i,j])
    return np.linalg.norm(I - AS) / 250000

def process_polar_parameters(path_folder, path_process):
    """
    This function takes the Raw polarimetric images and process them in order to get the four intensities and the
    polarimetric parameters such as the Stokes parameters, the Degree and the Angle of Polarization

    :param path_folder: The path to the folder containing all the Raw polarimetric images
    :param path_process: The path where to put all the polarimetric parameters
    """

    imgs_polar = sorted(os.listdir(path_folder))
    for k in range(len(imgs_polar)):

        # Intensities of the polarimetric image
        I, image = get_intensities(imgs_polar, k, path_process)

        # Stokes parameters

        Stokes = get_stokes_parameters(I, imgs_polar, k, path_process)

        i_as = []
        theta = np.arange(0, 0.0002, 0.000001)
        for t in theta:
            A = np.array([[1, np.cos(2*(t*np.pi/180)), np.sin(2*(t*np.pi/180))],
                          [1, np.cos(2*((t+45)*np.pi/180)), np.sin(2*((t+45)*np.pi/180))],
                          [1, np.cos(2*((t+90)*np.pi/180)), np.sin(2*((t+90)*np.pi/180))],
                          [1, np.cos(2*((t+135)*np.pi/180)), np.sin(2*((t+135)*np.pi/180))]])
            ias = computing_I_AS(I, Stokes, A)
            i_as.append(ias)

        plt.figure()
        plt.plot(theta, i_as, '+')
        plt.title("I - AS with the variation of theta")
        plt.savefig(str(k) + "variation.png")

path_folder = "/media/rblin/EC42-B858/test_polar_2/Raw/"
path_process = "/media/rblin/EC42-B858/test_polar_2/Process/"

process_polar_parameters(path_folder, path_process)