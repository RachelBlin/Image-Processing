import os
import imageio
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt
import cv2

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

    I[0] = image[0:image.shape[0]:2, 1:image.shape[1]:2] / np.max(image[0:image.shape[0]:2, 1:image.shape[1]:2]) * 255 # I0
    I[1] = image[0:image.shape[0]:2, 0:image.shape[1]:2] / np.max(image[0:image.shape[0]:2, 0:image.shape[1]:2]) * 255  # I45
    I[2] = image[1:image.shape[0]:2, 0:image.shape[1]:2] / np.max(image[1:image.shape[0]:2, 0:image.shape[1]:2]) * 255  # I90
    I[3] = image[1:image.shape[0]:2, 1:image.shape[1]:2] / np.max(image[1:image.shape[0]:2, 1:image.shape[1]:2]) * 255  # I135

    # Saving intensities
    if not os.path.exists(path_process + "I/"):
        os.mkdir(path_process + "I/")
    imageio.imwrite(path_process + "I/" + imgs_polar[k].split(".")[0] + "_I0.png", I[0])
    imageio.imwrite(path_process + "I/" + imgs_polar[k].split(".")[0] + "_I45.png", I[1])
    imageio.imwrite(path_process + "I/" + imgs_polar[k].split(".")[0] + "_I90.png", I[2])
    imageio.imwrite(path_process + "I/" + imgs_polar[k].split(".")[0] + "_I135.png", I[3])

    I.astype(int)

    return I, image

def get_intensities_generate(imgs_polar, k):
    """
        Split a Raw polarimetric images into the four intensities

        :param image: The image to be processed
        :param imgs_polar: The list containing all the polarimetric images in the folder
        :param k: The index of the image in the imgs_polar list
        :return: A vector containing all the intensities
        """
    image = imageio.imread(path_folder + "/" + imgs_polar[k])

    I = np.zeros((4, image.shape[0], image.shape[1]))

    I[0] = image[:, :, 0]
    I[1] = image[:, :, 1]
    I[2] = image[:, :, 2]
    I[3] = image[:, :, 3]

    # Saving intensities
    if not os.path.exists(path_process + "I/"):
        os.mkdir(path_process + "I/")
    imageio.imwrite(path_process + "I/" + imgs_polar[k].split(".")[0] + "_I0.png", I[0])
    imageio.imwrite(path_process + "I/" + imgs_polar[k].split(".")[0] + "_I45.png", I[1])
    imageio.imwrite(path_process + "I/" + imgs_polar[k].split(".")[0] + "_I90.png", I[2])
    imageio.imwrite(path_process + "I/" + imgs_polar[k].split(".")[0] + "_I135.png", I[3])

    I.astype(int)

    return I, image

def get_stokes_parameters(I, imgs_polar, k):
    """
    Compute the Stokes parameters

    :param I: The intensities of the image
    :param imgs_polar: The list containing all the polarimetric images in the folder
    :param k: The index of the image in the imgs_polar list
    :return: A vector containing the Stokes parameters
    """

    Stokes = np.zeros((4, I[0].shape[0], I[0].shape[1]), dtype=int)

    Stokes[0] = I[0] + I[2]
    Stokes[1] = I[0] - I[2]
    Stokes[2] = I[1] - I[3]
    Stokes[3] = I[1] + I[3]

    Stokes[0] = Stokes[0]
    Stokes[1] = Stokes[1]
    Stokes[2] = Stokes[2]
    Stokes[3] = Stokes[3]

    if not os.path.exists(path_process + "Stokes0/"):
        os.mkdir(path_process + "Stokes0/")
    imageio.imwrite(path_process + "Stokes0/" + imgs_polar[k].split(".")[0] + "_S0.png", Stokes[0])
    if not os.path.exists(path_process + "Stokes1/"):
        os.mkdir(path_process + "Stokes1/")
    imageio.imwrite(path_process + "Stokes1/" + imgs_polar[k].split(".")[0] + "_S1.png", Stokes[1])
    if not os.path.exists(path_process + "Stokes2/"):
        os.mkdir(path_process + "Stokes2/")
    imageio.imwrite(path_process + "Stokes2/" + imgs_polar[k].split(".")[0] + "_S2.png", Stokes[2])
    #imageio.imwrite(path_process + "Stokes/" + imgs_polar[k].split(".")[0] + "_Sdiff.png", abs(Stokes[3]-Stokes[0]))

    Stokes.astype(int)

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
    #phi = 0.5 * np.arctan2(Stokes[2], Stokes[1])
    DOP = np.zeros((I.shape[1], I.shape[2]))#, dtype=int)
    #rho = np.zeros((I.shape[1], I.shape[2]))
    l = 0
    for i in range(I.shape[1]):
        for j in range(I.shape[2]):
            #if np.divide(np.sqrt(np.square(Stokes[2, i, j]) + np.square(Stokes[1, i, j])), Stokes[0, i, j]) > 1:
                #l += 1
            if Stokes[0, i, j] == 0:
                Stokes[0, i, j] = 1
            DOP[i, j] = np.divide(np.sqrt(np.square(Stokes[2, i, j]) + np.square(Stokes[1, i, j])), Stokes[0, i, j])
            #rho[i, j] = np.divide(np.sqrt(np.square(Stokes[2, i, j]) + np.square(Stokes[1, i, j])), Stokes[0, i, j])
            """if DOP[i, j] == 0:
                equal_i.append([I[0, i, j], I[1, i, j], I[2, i, j], I[3, i, j]])"""

    #rho_one.append(l)

    DOP = DOP / np.max(DOP) * 255
    #rho = rho

    """im_cos = rho * np.cos(2*phi)
    im_cos = im_cos / np.max(im_cos) * 255
    im_sin = rho * np.sin(2*phi)
    im_sin = im_sin / np.max(im_sin) * 255

    # Saving the image in the format DOP*sin(AOP) and DOP*cos(AOP)

    if not os.path.exists(path_process + "Cos/"):
        os.mkdir(path_process + "Cos/")
    imageio.imwrite(path_process + "Cos/" + imgs_polar[k].split(".")[0] + "_cos.png", im_cos)
    if not os.path.exists(path_process + "Sin/"):
        os.mkdir(path_process + "Sin/")
    imageio.imwrite(path_process + "Sin/" + imgs_polar[k].split(".")[0] + "_sin.png", im_sin)"""

    # Saving the AOP and DOP

    if not os.path.exists(path_process + "AOP/"):
        os.mkdir(path_process + "AOP/")
    imageio.imwrite(path_process + "AOP/" + imgs_polar[k].split(".")[0] + "_AOP.png", AOP)
    if not os.path.exists(path_process + "DOP/"):
        os.mkdir(path_process + "DOP/")
    imageio.imwrite(path_process + "DOP/" + imgs_polar[k].split(".")[0] + "_DOP.png", DOP)

    #AOP.astype(int)
    #DOP.astype(int)

    return AOP, DOP#, im_cos, im_sin, rho, phi

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

    # Verifying that I0 + I90 = I45 + I135 by counting the number of pixels not respecting this constraint

    cons = 0
    for i in range(0, image.shape[1], 2):
        for j in range(0, image.shape[1], 2):
            if image[i, j + 1] + image[i + 1, j] != image[i, j] + image[i + 1, j + 1]:
                cons += 1


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
    print("Number of pixels for which I0 + I90 != I45 + I135: ", cons)
    print("Number of pixels for which S0 = 0: ", stokes_zero)
    print("Number of pixels for which DOP > 1: ", rho_one)
    #print("Values of the intensities leading to DOP=0: ", equal_i)

def fusion_algorithm(AOP, DOP, Stokes, I):
    # Computing the common component
    CC = np.minimum(np.minimum(AOP, DOP), Stokes[0])

    # Computing beta
    eps = 0.25
    alpha = 16
    beta = 2*AOP*((I[0]**eps-I[2]**eps+alpha)**(1/eps)+(I[1]**eps-I[3]**eps+alpha)**(1/eps))/(I[0]**eps+I[1]**eps+I[2]**eps+I[3]**eps)

    # Computing contribution of each plarized image
    modified_CC = np.minimum(np.minimum(beta, DOP), Stokes[0])
    S0_star = Stokes[0] - modified_CC
    DOP_star = DOP - modified_CC
    beta_star = beta - modified_CC

    # Removing the computed polarized contributions for each original polarized image
    S0_starstar = Stokes[0] - DOP_star - beta_star
    DOP_starstar = DOP - S0_star - beta_star
    beta_starstar = beta - DOP_star - S0_star

    # Fusing image
    hsv = np.zeros((I.shape[1], I.shape[2], 3))
    hsv[:,:,0] = S0_starstar
    hsv[:,:,1] = DOP_starstar
    hsv[:,:,2] = beta_starstar

    return hsv, CC, modified_CC, S0_star, DOP_star, beta, beta_star, S0_starstar, DOP_starstar, beta_starstar

def fusion_algorithm2(Stokes):
    S1_pr = Stokes[1]/Stokes[0]
    S2_pr = Stokes[2]/Stokes[0]

    mu_s1 = np.mean(S1_pr)
    mu_s2 = np.mean(S2_pr)

    sig_s1 = np.mean(S1_pr)
    sig_s2 = np.mean(S2_pr)

    enhanced_DOLP = np.sqrt(((S1_pr-mu_s1)/sig_s1)**2+((S2_pr-mu_s2)/sig_s2)**2)

    return enhanced_DOLP

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
        #I, image = get_intensities_generate(imgs_polar, k)

        # Stokes parameters

        Stokes = get_stokes_parameters(I, imgs_polar, k)


        # Angle (AOP) and Degree (DOP) of Polarization

        #AOP, DOP, im_cos, im_sin, rho, phi = get_params(I, Stokes, imgs_polar, k, rho_one, equal_i)
        AOP, DOP = get_params(I, Stokes, imgs_polar, k, rho_one, equal_i)

        """hsv, CC, modified_CC, S0_star, DOP_star, beta, beta_star, S0_starstar, DOP_starstar, beta_starstar = fusion_algorithm(phi, rho, Stokes, I)
        if not os.path.exists(path_process + "hsv_fusion/"):
            os.mkdir(path_process + "hsv_fusion/")
        imageio.imwrite(path_process + "hsv_fusion/" + imgs_polar[k].split(".")[0] + ".png", hsv)
        imageio.imwrite(path_process + "hsv_fusion/" + imgs_polar[k].split(".")[0] + "_CC.png", CC)
        imageio.imwrite(path_process + "hsv_fusion/" + imgs_polar[k].split(".")[0] + "_modifCC.png", modified_CC)
        imageio.imwrite(path_process + "hsv_fusion/" + imgs_polar[k].split(".")[0] + "_S0star.png", S0_star)
        imageio.imwrite(path_process + "hsv_fusion/" + imgs_polar[k].split(".")[0] + "_DOPstar.png", DOP_star)
        imageio.imwrite(path_process + "hsv_fusion/" + imgs_polar[k].split(".")[0] + "_beta.png", beta)
        imageio.imwrite(path_process + "hsv_fusion/" + imgs_polar[k].split(".")[0] + "_betastar.png", beta_star)
        imageio.imwrite(path_process + "hsv_fusion/" + imgs_polar[k].split(".")[0] + "_S0starstar.png", S0_starstar)
        imageio.imwrite(path_process + "hsv_fusion/" + imgs_polar[k].split(".")[0] + "_DOPstarstar.png", DOP_starstar)
        imageio.imwrite(path_process + "hsv_fusion/" + imgs_polar[k].split(".")[0] + "_betastarstar.png", beta_starstar)

        enhanced_DOLP = fusion_algorithm2(Stokes)
        if not os.path.exists(path_process + "enhanced_dolp/"):
            os.mkdir(path_process + "enhanced_dolp/")
        imageio.imwrite(path_process + "enhanced_dolp/" + imgs_polar[k].split(".")[0] + ".png", enhanced_DOLP)

        # Max and Min fusion of AOP, DOP, I0, S1 an S2

        Max = np.zeros((I.shape[1], I.shape[2]))
        for i in range(I.shape[1]):
            for j in range(I.shape[2]):
                Max[i,j] = max(DOP[i,j], AOP[i,j], Stokes[1,i,j], Stokes[2,i,j], I[0,i,j])

        if not os.path.exists(path_process + "Max_fusion/"):
            os.mkdir(path_process + "Max_fusion/")
        imageio.imwrite(path_process + "Max_fusion/" + imgs_polar[k].split(".")[0] + "_max.png", Max)

        Min = np.zeros((I.shape[1], I.shape[2]))
        for i in range(I.shape[1]):
            for j in range(I.shape[2]):
                Min[i, j] = min(DOP[i, j], AOP[i, j], Stokes[1, i, j], Stokes[2, i, j], I[0, i, j])

        if not os.path.exists(path_process + "Min_fusion/"):
            os.mkdir(path_process + "Min_fusion/")
        imageio.imwrite(path_process + "Min_fusion/" + imgs_polar[k].split(".")[0] + "_min.png", Min)

        # Verifying the polarimetric constraints

        #polarimetric_constraints(I, image, Stokes, imgs_polar, k, stokes_zero, rho_one, equal_i)"""

        #concatenate_frames(I, Stokes, AOP, DOP, path_process, k, imgs_polar) #, Min, Max, im_cos, im_sin, rho, phi)

def concatenate_frames(I, Stokes, AOP, DOP, path_process, k, imgs_polar): #, Min, Max, im_cos, im_sin, rho, phi):
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

    """# Fusion
    im_fusion = np.zeros((500, 500, 5), dtype=int)
    im_fusion[:, :, 0] = Stokes[0]
    im_fusion[:, :, 1] = Stokes[1]
    im_fusion[:, :, 2] = Stokes[2]
    im_fusion[:, :, 3] = AOP
    im_fusion[:, :, 4] = DOP
    if not os.path.exists(path_process + "Fusion/"):
        os.mkdir(path_process + "Fusion/")
    np.save(path_process + "Fusion/" + imgs_polar[k].split(".")[0], im_fusion.astype(np.uint8))"""

    """# RetinaNet intensities
    im_I04590 = np.zeros((500, 500, 3))
    im_I04590[:, :, 0] = I[0]
    im_I04590[:, :, 1] = I[1]
    im_I04590[:, :, 2] = I[2]
    if not os.path.exists(path_process + "I04590/"):
        os.mkdir(path_process + "I04590/")
    imageio.imwrite(path_process + "I04590/" + imgs_polar[k].split(".")[0] + ".png", im_I04590)

    # Min Max total intensity
    im_min_max = np.zeros((500, 500, 3))
    im_min_max[:, :, 0] = Stokes[0]
    im_min_max[:, :, 1] = Max
    im_min_max[:, :, 2] = Min
    if not os.path.exists(path_process + "MinMax/"):
        os.mkdir(path_process + "MinMax/")
    imageio.imwrite(path_process + "MinMax/" + imgs_polar[k].split(".")[0] + ".png", im_min_max)

    # Cos Sin total intensity
    im_cos_sin = np.zeros((500, 500, 3))
    im_cos_sin[:, :, 0] = Stokes[0]
    im_cos_sin[:, :, 1] = im_cos
    im_cos_sin[:, :, 2] = im_sin
    if not os.path.exists(path_process + "CosSin/"):
        os.mkdir(path_process + "CosSin/")
    imageio.imwrite(path_process + "CosSin/" + imgs_polar[k].split(".")[0] + ".png", im_cos_sin)"""

    """# Cos Sin total intensity
    im_cos_sin = np.zeros((500, 500, 3))
    im_cos_sin[:, :, 0] = DOP
    im_cos_sin[:, :, 1] = im_cos
    im_cos_sin[:, :, 2] = im_sin
    if not os.path.exists(path_process + "CosSin2_s/"):
        os.mkdir(path_process + "CosSin2_s/")
    imageio.imwrite(path_process + "CosSin2_s/" + imgs_polar[k].split(".")[0] + ".png", im_cos_sin)"""


    """im_I045135 = np.zeros((500, 500, 3))
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
    imageio.imwrite(path_process + "RetinaNet_eq10/" + str(k) + ".png", im_I4590135)"""

    # retinaNet Stokes
    im_Stokes = np.zeros((Stokes.shape[1], Stokes.shape[2], 3))
    im_Stokes[:, :, 0] = Stokes[0]
    im_Stokes[:, :, 1] = Stokes[1]
    im_Stokes[:, :, 2] = Stokes[2]
    if not os.path.exists(path_process + "Stokes/"):
        os.mkdir(path_process + "Stokes/")
    imageio.imwrite(path_process + "Stokes/" + imgs_polar[k].split(".")[0] + ".png", im_Stokes)
    """

    # RetinaNet Params
    im_Params = np.zeros((500, 500, 3))
    im_Params[:, :, 0] = Stokes[0]
    im_Params[:, :, 1] = AOP
    im_Params[:, :, 2] = DOP
    if not os.path.exists(path_process + "Params/"):
        os.mkdir(path_process + "Params/")
    imageio.imwrite(path_process + "Params/" + imgs_polar[k].split(".")[0] + ".png", im_Params)"""

    """# HSV image
    HSV = np.zeros((500, 500, 3))
    HSV[:, :, 0] = AOP / 255 * 179
    HSV[:, :, 1] = DOP
    HSV[:, :, 2] = Stokes[0]
    if not os.path.exists(path_process + "HSV/"):
        os.mkdir(path_process + "HSV/")
    imageio.imwrite(path_process + "HSV/" + imgs_polar[k].split(".")[0] + ".png", HSV)"""

    """inten = (I[0] + I[1] + I[2] + I[3]) / 2

    hsv = np.uint8(cv2.merge(((phi + np.pi/2)/np.pi*180,rho/np.max(rho)*255, inten/inten.max()*255)))
    if not os.path.exists(path_process + "HSV_2/"):
        os.mkdir(path_process + "HSV_2/")
    imageio.imwrite(path_process + "HSV_2/" + imgs_polar[k].split(".")[0] + ".png", hsv)"""

    """# TSV image
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
    imageio.imwrite(path_process + "RetinaNet_Pauli/" + str(k) + ".png", Pauli)"""

    """Pauli = np.zeros((500, 500, 3))
    Pauli[:, :, 0] = I[0] + I[2]
    Pauli[:, :, 1] = I[1]
    Pauli[:, :, 2] = I[0] - I[2]
    if not os.path.exists(path_process + "Pauli2/"):
        os.mkdir(path_process + "Pauli2/")
    imageio.imwrite(path_process + "Pauli2/" + imgs_polar[k].split(".")[0] + ".png", Pauli)"""

    """Pauli = np.zeros((500, 500, 3))
    Pauli[:, :, 0] = I[0] + I[2]
    Pauli[:, :, 1] = I[1]
    Pauli[:, :, 2] = I[0] - I[2]
    if not os.path.exists(path_process + "Pauli2_inv/"):
        os.mkdir(path_process + "Pauli2_inv/")
    imageio.imwrite(path_process + "Pauli2_inv/" + imgs_polar[k].split(".")[0] + ".png", Pauli)"""

    """Pauli = np.zeros((500, 500, 3))
    Pauli[:, :, 0] = Stokes[0]
    Pauli[:, :, 1] = I[1]
    Pauli[:, :, 2] = Stokes[1]
    if not os.path.exists(path_process + "Pauli2/"):
        os.mkdir(path_process + "Pauli2/")
    imageio.imwrite(path_process + "Pauli2/" + imgs_polar[k].split(".")[0] + ".png", Pauli)

    Pauli = np.zeros((500, 500, 3))
    Pauli[:, :, 0] = I[0]
    Pauli[:, :, 1] = (I[1]+I[3])/2
    Pauli[:, :, 2] = I[2]
    if not os.path.exists(path_process + "Sinclair/"):
        os.mkdir(path_process + "Sinclair/")
    imageio.imwrite(path_process + "Sinclair/" + imgs_polar[k].split(".")[0] + ".png", Pauli)

    Pauli = np.zeros((500, 500, 3))
    Pauli[:, :, 0] = Stokes[0]
    Pauli[:, :, 1] = I[1] + I[3]
    Pauli[:, :, 2] = Stokes[1]
    if not os.path.exists(path_process + "Pauli/"):
        os.mkdir(path_process + "Pauli/")
    imageio.imwrite(path_process + "Pauli/" + imgs_polar[k].split(".")[0] + ".png", Pauli)

    Pauli = np.zeros((500, 500, 3))
    Pauli[:, :, 0] = I[0]
    Pauli[:, :, 1] = I[2]
    Pauli[:, :, 2] = DOP
    if not os.path.exists(path_process + "Test/"):
        os.mkdir(path_process + "Test/")
    imageio.imwrite(path_process + "Test/" + imgs_polar[k].split(".")[0] + ".png", Pauli)

    Pauli = np.zeros((500, 500, 3))
    Pauli[:, :, 0] = I[1]
    Pauli[:, :, 1] = I[3]
    Pauli[:, :, 2] = DOP
    if not os.path.exists(path_process + "Test1/"):
        os.mkdir(path_process + "Test1/")
    imageio.imwrite(path_process + "Test1/" + imgs_polar[k].split(".")[0] + ".png", Pauli)

    Pauli = np.zeros((500, 500, 3))
    Pauli[:, :, 0] = I[0]
    Pauli[:, :, 1] = I[3]
    Pauli[:, :, 2] = DOP
    if not os.path.exists(path_process + "Test2/"):
        os.mkdir(path_process + "Test2/")
    imageio.imwrite(path_process + "Test2/" + imgs_polar[k].split(".")[0] + ".png", Pauli)

    Pauli = np.zeros((500, 500, 3))
    Pauli[:, :, 0] = I[0]
    Pauli[:, :, 1] = I[1] + I[2] - I[3]
    Pauli[:, :, 2] = DOP
    if not os.path.exists(path_process + "Test3/"):
        os.mkdir(path_process + "Test3/")
    imageio.imwrite(path_process + "Test3/" + imgs_polar[k].split(".")[0] + ".png", Pauli)"""

    """Pauli = np.zeros((500, 500, 3))
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
    imageio.imwrite(path_process + "RetinaNet_Rachel2/" + str(k) + ".png", Rachel)"""

#path_folder = "/home/rblin/Documents/brouillard_statistiques/real_raw/POLAR/"
#path_process = "/home/rblin/Documents/brouillard_statistiques/real_raw/PARAM_POLAR/"

path_folder = "//home/rblin/Documents/New_illustrations_ACCV/POLAR/"
path_process = "//home/rblin/Documents/New_illustrations_ACCV/PARAM_POLAR/"

process_polar_parameters(path_folder, path_process)