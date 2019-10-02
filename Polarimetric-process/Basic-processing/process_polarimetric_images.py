import os
import imageio
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt

# Image composée de superpixels et chaque super pixel est composé de la façon suivante :
#  45 | 0
# ---------
# 90  | 135
# => pour les images de la caméra polarimétrique

def get_intensities(imgs_polar, k):
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

    # Saving intensities
    if not os.path.exists(path_process + "I/"):
        os.mkdir(path_process + "I/")
    imageio.imwrite(path_process + "I/" + imgs_polar[k].split(".")[0] + "_I0.png", I[0])
    imageio.imwrite(path_process + "I/" + imgs_polar[k].split(".")[0] + "_I45.png", I[1])
    imageio.imwrite(path_process + "I/" + imgs_polar[k].split(".")[0] + "_I90.png", I[2])
    imageio.imwrite(path_process + "I/" + imgs_polar[k].split(".")[0] + "_I135.png", I[3])

    return I, image

def get_stokes_parameters(I, imgs_polar, k):
    """
    Compute the Stokes parameters

    :param I: The intensities of the image
    :param imgs_polar: The list containing all the polarimetric images in the folder
    :param k: The index of the image in the imgs_polar list
    :return: A vector containing the Stokes parameters
    """

    Stokes = np.zeros((4, I[0].shape[0], I[0].shape[1]))

    Stokes[0] = I[0] + I[2]
    Stokes[1] = I[0] - I[2]
    Stokes[2] = I[1] - I[3]

    if not os.path.exists(path_process + "Stokes/"):
        os.mkdir(path_process + "Stokes/")
    imageio.imwrite(path_process + "Stokes/" + imgs_polar[k].split(".")[0] + "_S0.png", Stokes[0])
    imageio.imwrite(path_process + "Stokes/" + imgs_polar[k].split(".")[0] + "_S1.png", Stokes[1])
    imageio.imwrite(path_process + "Stokes/" + imgs_polar[k].split(".")[0] + "_S2.png", Stokes[2])

    return Stokes

def get_params(I, Stokes, imgs_polar, k, rho_one, equal_i):
    """
    A function to get the AOP and DOP of an image

    :param I : The intensities of the image
    :param Stokes: The Stokes parameters of an image
    :param imgs_polar: The list containing all the polarimetric images in the folder
    :param k: The index of the image in the imgs_polar list
    :param rho_one: The number of values per image for which DOP > 1
    :param equal_i: The value of the intensities leading to DOP = 0
    :return: The Angle and Degree of Polarization
    """

    AOP = (0.5 * np.arctan2(Stokes[2], Stokes[1]) + np.pi / 2) / np.pi * 255
    phi = 0.5 * np.arctan2(Stokes[2], Stokes[1])
    DOP = np.zeros((500, 500))
    rho = np.zeros((500, 500))
    l = 0
    for i in range(500):
        for j in range(500):
            if np.divide(np.sqrt(np.square(Stokes[2, i, j]) + np.square(Stokes[1, i, j])), Stokes[0, i, j]) > 1:
                l += 1
            DOP[i, j] = np.divide(np.sqrt(np.square(Stokes[2, i, j]) + np.square(Stokes[1, i, j])), Stokes[0, i, j])
            rho[i, j] = np.divide(np.sqrt(np.square(Stokes[2, i, j]) + np.square(Stokes[1, i, j])), Stokes[0, i, j])
            if DOP[i, j] == 0:
                equal_i.append([I[0, i, j], I[1, i, j], I[2, i, j], I[3, i, j]])

    rho_one.append(l)

    DOP = DOP / np.max(DOP) * 255
    rho = rho / np.max(rho) * 255

    im_cos = rho * np.cos(phi)
    im_cos = im_cos / np.max(im_cos) * 255
    im_sin = rho * np.sin(phi)
    im_sin = im_sin / np.max(im_sin) * 255

    # Saving the image in the format DOP*sin(AOP) and DOP*cos(AOP)

    if not os.path.exists(path_process + "CosSin/"):
        os.mkdir(path_process + "CosSin/")
    imageio.imwrite(path_process + "CosSin/" + imgs_polar[k].split(".")[0] + "_cos.png", im_cos)
    imageio.imwrite(path_process + "CosSin/" + imgs_polar[k].split(".")[0] + "_sin.png", im_sin)

    # Saving the AOP and DOP

    if not os.path.exists(path_process + "Params/"):
        os.mkdir(path_process + "Params/")
    imageio.imwrite(path_process + "Params/" + imgs_polar[k].split(".")[0] + "_AOP.png", AOP)
    imageio.imwrite(path_process + "Params/" + imgs_polar[k].split(".")[0] + "_DOP.png", DOP)

    return AOP, DOP

def plot_histogram(bins, data, title, saving_path, hist_name):
    """
    A function to plot histograms

    :param bins: The ranges of the bars
    :param data: The data to be plotted
    :param title: The title of the histogram
    :param saving_path: The path of the folder to save the histogram to
    :param hist_name: The name of the figure
    """

    x = np.asarray(data)
    plt.figure()
    plt.hist(x[np.isfinite(x)], bins)
    plt.title(title)
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)
    plt.savefig(saving_path + hist_name)

def polarimetric_constraints(I, image, Stokes, imgs_polar, k, stokes_zero, rho_one, equal_i):
    """
    A function to plot several graphics to estimate the noise in polarimetric images

    This function aims to verify the different polarimetric constraints in order to estimate the noise
    :param I: The intensities of the image
    :param image: The image to be processed
    :param Stokes: The Stokes parameters of an image
    :param imgs_polar: The list containing all the polarimetric images in the folder
    :param k: The index of the image in the imgs_polar list
    :param stokes_zero: The number of pixels for which S0 = 0 in the image
    :param rho_one: The number of values per image for which DOP > 1
    :param equal_i: The value of the intensities leading to DOP = 0
    """

    # Verifying the polarimetric constraint I0 + I90 = I45 + I135 by plotting an histogram of the ratio
    # (I0 + I90)/(I45 + I135) for each pixel of the image
    intens = []
    for i in range(0, image.shape[1], 2):
        for j in range(0, image.shape[1], 2):
            intens.append((image[i, j + 1] + image[i + 1, j]) / (image[i, j] + image[i + 1, j + 1]))
    bins = [0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9,
            0.925, 0.95, 0.975, 1, 1.025, 1.05, 1.075, 1.1, 1.125, 1.15, 1.175, 1.2]
    title = "(I0 + I90)/(I45 + I135)"
    saving_path = path_process + "hist/"
    hist_name = imgs_polar[k].split(".")[0] + "_hist.png"
    plot_histogram(bins, intens, title, saving_path, hist_name)

    # Verifying the polarimetric constraint I = AS by plotting an histogram of ||I-AS||/(||I|| + ||AS||) for each
    # pixel of the image
    i_as = []
    i_delta_as = []
    for i in range(500):
        for j in range(500):
            I_temp = np.array([I[0, i, j], I[1, i, j], I[2, i, j], I[3, i, j]])
            AS = np.array([0.5 * (Stokes[0, i, j] + Stokes[1, i, j]), 0.5 * (Stokes[0, i, j] + Stokes[2, i, j]),
                           0.5 * (Stokes[0, i, j] - Stokes[1, i, j]), 0.5 * (Stokes[0, i, j] - Stokes[2, i, j])])
            i_as.append(np.linalg.norm(I_temp - AS) / (np.linalg.norm(I_temp) + np.linalg.norm(AS)))
            delta_AS = np.array(
                [0.5 * (Stokes[0, i, j] + Stokes[1, i, j]), 0.5 * (Stokes[0, i, j] + 1.125 * Stokes[2, i, j]),
                 0.5 * (Stokes[0, i, j] - Stokes[1, i, j]), 0.5 * (Stokes[0, i, j] - 0.875 * Stokes[2, i, j])])
            i_delta_as.append(np.linalg.norm(I_temp - delta_AS) / (np.linalg.norm(I_temp) + np.linalg.norm(delta_AS)))

    bins = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
    title = "||I-AS||/(||I|| + ||AS||)"
    hist_name = imgs_polar[k].split(".")[0] + "_hist_ias.png"
    plot_histogram(bins, i_as, title, saving_path, hist_name)
    title = "||I-AS||/(||I|| + ||AS||) avec A = A + delta(A)"
    hist_name = imgs_polar[k].split(".")[0] + "_hist_ias_delta.png"
    plot_histogram(bins, i_delta_as, title, saving_path, hist_name)

    z = np.count_nonzero(Stokes[0] == 0)

    stokes_zero.append(z)

    # Verifying the following polarimetric constraints in order to evaluate the noise:
    #   - DOP in [0, 1]
    #   - S0 > 0
    #   - I0 = I45 = I90 = I135 when DOP is 0
    print("Ordered names of the processed polarimetric images: ", imgs_polar)
    print("Number of pixels for which S0 = 0: ", stokes_zero)
    print("Number of pixels for which DOP > 1: ", rho_one)
    print("Values of the intensities leading to DOP=0: ", equal_i)


def process_polar_parameters(path_folder, path_process):
    """
    This function takes the Raw polarimetric images and process them in order to get the four intensities and the
    polarimetric parameters such as the Stokes parameters, the Degree and the Angle of Polarization

    :param path_folder: The path to the folder containing all the Raw polarimetric images
    :param path_process: The path where to put all the polarimetric parameters
    """

    imgs_polar = sorted(os.listdir(path_folder))
    stokes_zero = []
    rho_one = []
    equal_i = []
    for k in range(len(imgs_polar)):

        # Intensities of the polarimetric image
        I, image = get_intensities(imgs_polar, k)

        # Stokes parameters

        Stokes = get_stokes_parameters(I, imgs_polar, k)


        # Angle (AOP) and Degree (DOP) of Polarization

        AOP, DOP = get_params(I, Stokes, imgs_polar, k, rho_one, equal_i)

        # Max and Min fusion of AOP, DOP, I0, S1 an S2

        Max = np.zeros((500, 500))
        for i in range(500):
            for j in range(500):
                Max[i,j] = max(DOP[i,j], AOP[i,j], Stokes[1,i,j], Stokes[2,i,j], I[0,i,j])

        if not os.path.exists(path_process + "Max_fusion/"):
            os.mkdir(path_process + "Max_fusion/")
        imageio.imwrite(path_process + "Max_fusion/" + imgs_polar[k].split(".")[0] + "_max.png", Max)

        Min = np.zeros((500, 500))
        for i in range(500):
            for j in range(500):
                Min[i, j] = min(DOP[i, j], AOP[i, j], Stokes[1, i, j], Stokes[2, i, j], I[0, i, j])

        if not os.path.exists(path_process + "Min_fusion/"):
            os.mkdir(path_process + "Min_fusion/")
        imageio.imwrite(path_process + "Min_fusion/" + imgs_polar[k].split(".")[0] + "_min.png", Min)

        # Verifying the polarimetric constraints

        polarimetric_constraints(I, image, Stokes, imgs_polar, k, stokes_zero, rho_one, equal_i)

        #concatenate_frames(I, Stokes, AOP, DOP, path_process, k, imgs_polar)

def concatenate_frames(I, Stokes, AOP, DOP, path_process, k, imgs_polar):
    """
    A function to concatenate frames in order to get 3 channels images

    :param I: The intensities of the image
    :param Stokes: The Stokes parameters of an image
    :param AOP: The Angle op Polarization of the image
    :param DOP: The Degree of Polarization of the image
    :param path_process: The path where to put all the polarimetric parameters
    :param k: The index of the image in the imgs_polar list
    :param imgs_polar: The list containing all the polarimetric images in the folder
    """

    # RetinaNet intensities
    im_I04590 = np.zeros((500, 500, 3))
    im_I04590[:, :, 0] = I[0]
    im_I04590[:, :, 1] = I[1]
    im_I04590[:, :, 2] = I[2]
    if not os.path.exists(path_process + "I04590/"):
        os.mkdir(path_process + "I04590/")
    imageio.imwrite(path_process + "I04590/" + imgs_polar[k].split(".")[0] + ".png", im_I04590)

    im_I045135 = np.zeros((500, 500, 3))
    im_I045135[:, :, 0] = I[0]
    im_I045135[:, :, 1] = I[3]
    im_I045135[:, :, 2] = I[1]
    if not os.path.exists(path_process + "I013545/"):
        os.mkdir(path_process + "I013545/")
    imageio.imwrite(path_process + "I013545/" + imgs_polar[k].split(".")[0] + ".png", im_I045135)

    im_I090135 = np.zeros((500, 500, 3))
    im_I090135[:, :, 0] = I[0]
    im_I090135[:, :, 1] = I[2]
    im_I090135[:, :, 2] = I[3]
    if not os.path.exists(path_process + "I090135/"):
        os.mkdir(path_process + "I090135/")
    imageio.imwrite(path_process + "I090135/" + imgs_polar[k].split(".")[0] + ".png", im_I090135)

    im_I4590135 = np.zeros((500, 500, 3))
    im_I4590135[:, :, 0] = I[1]
    im_I4590135[:, :, 1] = I[2]
    im_I4590135[:, :, 2] = I[3]
    if not os.path.exists(path_process + "I4590135/"):
        os.mkdir(path_process + "I4590135/")
    imageio.imwrite(path_process + "I4590135/" + imgs_polar[k].split(".")[0] + ".png", im_I4590135)

    im_I090135 = np.zeros((500, 500, 3))
    im_I090135[:, :, 0] = I[0] - I[1]
    im_I090135[:, :, 1] = I[0]
    im_I090135[:, :, 2] = I[0] + I[1]
    if not os.path.exists(path_process + "RetinaNet_Ieq1/"):
        os.mkdir(path_process + "RetinaNet_Ieq1/")
    imageio.imwrite(path_process + "RetinaNet_Ieq1/" + str(k) + ".png", im_I090135)

    im_I090135 = np.zeros((500, 500, 3))
    im_I090135[:, :, 0] = I[0] - I[3]
    im_I090135[:, :, 1] = I[0]
    im_I090135[:, :, 2] = I[0] + I[3]
    if not os.path.exists(path_process + "RetinaNet_Ieq2/"):
        os.mkdir(path_process + "RetinaNet_Ieq2/")
    imageio.imwrite(path_process + "RetinaNet_Ieq2/" + str(k) + ".png", im_I090135)

    im_I090135 = np.zeros((500, 500, 3))
    im_I090135[:, :, 0] = I[1] - I[2]
    im_I090135[:, :, 1] = I[1]
    im_I090135[:, :, 2] = I[1] + I[2]
    if not os.path.exists(path_process + "RetinaNet_Ieq3/"):
        os.mkdir(path_process + "RetinaNet_Ieq3/")
    imageio.imwrite(path_process + "RetinaNet_Ieq3/" + str(k) + ".png", im_I090135)

    im_I090135 = np.zeros((500, 500, 3))
    im_I090135[:, :, 0] = I[0]/I[1]
    im_I090135[:, :, 1] = I[0]/I[2]
    im_I090135[:, :, 2] = I[0]/I[3]
    if not os.path.exists(path_process + "RetinaNet_Ieq4/"):
        os.mkdir(path_process + "RetinaNet_Ieq4/")
    imageio.imwrite(path_process + "RetinaNet_Ieq4/" + str(k) + ".png", im_I090135)

    im_I4590135 = np.zeros((500, 500, 3))
    im_I4590135[:, :, 0] = I[0]
    im_I4590135[:, :, 1] = I[0]/I[1]
    im_I4590135[:, :, 2] = I[0]/I[2]
    if not os.path.exists(path_process + "RetinaNet_eq5/"):
        os.mkdir(path_process + "RetinaNet_eq5/")
    imageio.imwrite(path_process + "RetinaNet_eq5/" + str(k) + ".png", im_I4590135)

    im_I4590135 = np.zeros((500, 500, 3))
    im_I4590135[:, :, 0] = I[0]
    im_I4590135[:, :, 1] = I[0] / I[2]
    im_I4590135[:, :, 2] = I[0] / I[3]
    if not os.path.exists(path_process + "RetinaNet_eq6/"):
        os.mkdir(path_process + "RetinaNet_eq6/")
    imageio.imwrite(path_process + "RetinaNet_eq6/" + str(k) + ".png", im_I4590135)

    im_I4590135 = np.zeros((500, 500, 3))
    im_I4590135[:, :, 0] = I[1] / I[0]
    im_I4590135[:, :, 1] = I[1] / I[2]
    im_I4590135[:, :, 2] = I[1] / I[3]
    if not os.path.exists(path_process + "RetinaNet_eq7/"):
        os.mkdir(path_process + "RetinaNet_eq7/")
    imageio.imwrite(path_process + "RetinaNet_eq7/" + str(k) + ".png", im_I4590135)

    im_I4590135 = np.zeros((500, 500, 3))
    im_I4590135[:, :, 0] = I[2] / I[0]
    im_I4590135[:, :, 1] = I[2] / I[1]
    im_I4590135[:, :, 2] = I[2] / I[3]
    if not os.path.exists(path_process + "RetinaNet_eq8/"):
        os.mkdir(path_process + "RetinaNet_eq8/")
    imageio.imwrite(path_process + "RetinaNet_eq8/" + str(k) + ".png", im_I4590135)

    im_I4590135 = np.zeros((500, 500, 3))
    im_I4590135[:, :, 0] = I[3] / I[0]
    im_I4590135[:, :, 1] = I[3] / I[1]
    im_I4590135[:, :, 2] = I[3] / I[2]
    if not os.path.exists(path_process + "RetinaNet_eq9/"):
        os.mkdir(path_process + "RetinaNet_eq9/")
    imageio.imwrite(path_process + "RetinaNet_eq9/" + str(k) + ".png", im_I4590135)

    im_I4590135 = np.zeros((500, 500, 3))
    im_I4590135[:, :, 0] = I[0]/I[1]
    im_I4590135[:, :, 1] = I[0] / I[2]
    im_I4590135[:, :, 2] = DOP/255
    if not os.path.exists(path_process + "RetinaNet_eq10/"):
        os.mkdir(path_process + "RetinaNet_eq10/")
    imageio.imwrite(path_process + "RetinaNet_eq10/" + str(k) + ".png", im_I4590135)

    # retinaNet Stokes
    im_Stokes = np.zeros((500, 500, 3))
    im_Stokes[:, :, 0] = Stokes[0]
    im_Stokes[:, :, 1] = Stokes[1]
    im_Stokes[:, :, 2] = Stokes[2]
    if not os.path.exists(path_process + "Stokes/"):
        os.mkdir(path_process + "Stokes/")
    imageio.imwrite(path_process + "Stokes/" + imgs_polar[k].split(".")[0] + ".png", im_Stokes)

    # RetinaNet Params
    im_Params = np.zeros((500, 500, 3))
    im_Params[:, :, 0] = Stokes[0]
    im_Params[:, :, 1] = AOP
    im_Params[:, :, 2] = DOP
    if not os.path.exists(path_process + "Params/"):
        os.mkdir(path_process + "Params/")
    imageio.imwrite(path_process + "Params/" + imgs_polar[k].split(".")[0] + ".png", im_Params)

    # HSV image
    HSV = np.zeros((500, 500, 3))
    inten = (I[0] + I[1] + I[2] + I[3]) / 2
    HSV[:, :, 0] = AOP
    HSV[:, :, 1] = DOP
    HSV[:, :, 2] = Stokes[0]
    if not os.path.exists(path_process + "HSV/"):
        os.mkdir(path_process + "HSV/")
    imageio.imwrite(path_process + "HSV/" + imgs_polar[k].split(".")[0] + ".png", HSV)

    # TSV image
    TSV = np.zeros((500, 500, 3))
    TSV[:, :, 0] = AOP
    TSV[:, :, 1] = DOP
    TSV[:, :, 2] = inten / inten.max() * 255
    if not os.path.exists(path_process + "RetinaNet_TSV/"):
        os.mkdir(path_process + "RetinaNet_TSV/")
    imageio.imwrite(path_process + "RetinaNet_TSV/" + str(k) + ".png", TSV)

    # Pauli image
    Pauli = np.zeros((500, 500, 3))
    Pauli[:, :, 0] = I[2]
    Pauli[:, :, 1] = I[1]
    Pauli[:, :, 2] = I[0]
    if not os.path.exists(path_process + "RetinaNet_Pauli/"):
        os.mkdir(path_process + "RetinaNet_Pauli/")
    imageio.imwrite(path_process + "RetinaNet_Pauli/" + str(k) + ".png", Pauli)

    Pauli = np.zeros((500, 500, 3))
    Pauli[:, :, 0] = I[0] + I[2]
    Pauli[:, :, 1] = I[1]
    Pauli[:, :, 2] = I[0] - I[2]
    if not os.path.exists(path_process + "Pauli2/"):
        os.mkdir(path_process + "Pauli2/")
    imageio.imwrite(path_process + "Pauli2/" + imgs_polar[k].split(".")[0] + ".png", Pauli)

    Pauli = np.zeros((500, 500, 3))
    Pauli[:, :, 0] = I[0]
    Pauli[:, :, 1] = I[1]
    Pauli[:, :, 2] = (I[0]/I[1]) #/ np.amax(I[0] / I[1]) * 255
    if not os.path.exists(path_process + "Pauli3/"):
        os.mkdir(path_process + "Pauli3/")
    imageio.imwrite(path_process + "Pauli3/" + imgs_polar[k].split(".")[0] + ".png", Pauli)

    Rachel = np.zeros((500, 500, 3))
    Rachel[:, :, 0] = Stokes[0]
    Rachel[:, :, 1] = Stokes[1]
    Rachel[:, :, 2] = DOP
    if not os.path.exists(path_process + "RetinaNet_Rachel/"):
        os.mkdir(path_process + "RetinaNet_Rachel/")
    imageio.imwrite(path_process + "RetinaNet_Rachel/" + str(k) + ".png", Rachel)

    Rachel = np.zeros((500, 500, 3))
    Rachel[:, :, 0] = I[1]
    Rachel[:, :, 1] = I[0]
    Rachel[:, :, 2] = DOP
    if not os.path.exists(path_process + "RetinaNet_Rachel2/"):
        os.mkdir(path_process + "RetinaNet_Rachel2/")
    imageio.imwrite(path_process + "RetinaNet_Rachel2/" + str(k) + ".png", Rachel)

path_folder = "/media/rblin/EC42-B858/test_polar_2/Raw/"
path_process = "/media/rblin/EC42-B858/test_polar_2/Process/"

process_polar_parameters(path_folder, path_process)