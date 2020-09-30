import os
import imageio
import numpy as np
import matplotlib.pyplot as plt

def get_three_channels(path_folder, path_destination):
    imgs_polar = sorted(os.listdir(path_folder))
    for k in imgs_polar:
        image = imageio.imread(path_folder + "/" + k)
        I = np.zeros((image.shape[0], image.shape[1], 3))
        I[:,:,0] = image[:,:,0] # I0
        I[:,:,1] = image[:,:,1] # I45
        I[:,:,2] = image[:,:,2] # I90
        imageio.imwrite(path_destination + "/" + k, I)

def get_four_intensities(path_folder, path_destination):
    imgs_polar = sorted(os.listdir(path_folder))
    for k in imgs_polar:
        image = imageio.imread(path_folder + "/" + k)
        I = np.zeros((image.shape[0], image.shape[1], 4))
        I[:, :, 0] = image[:, :, 0]  # I0
        I[:, :, 1] = image[:, :, 1]  # I45
        I[:, :, 2] = image[:, :, 2]  # I90
        I[:, :, 3] = image[:, :, 3]  # I135
        imageio.imwrite(path_destination + "/" + "I0_" + k, I[:, :, 0])
        imageio.imwrite(path_destination + "/" + "I45_" + k, I[:, :, 1])
        imageio.imwrite(path_destination + "/" + "I90_" + k, I[:, :, 2])
        imageio.imwrite(path_destination + "/" + "I135_" + k, I[:, :, 3])

def plot_histogram_ias(path_folder):
    folder_list = sorted(os.listdir(path_folder))
    for f in folder_list:
        imgs_polar = sorted(os.listdir(path_folder + "/" + f))
        i_as = []
        stokes_param = []
        stokes_zero = []

        for k in imgs_polar:
            image = imageio.imread(path_folder + "/" + f + "/" + k)
            I = np.zeros((4, image.shape[0], image.shape[1]))
            I[0] = image[:, :, 0]  # I0
            I[1] = image[:, :, 1]  # I45
            I[2] = image[:, :, 2]  # I90
            I[3] = image[:, :, 3]  # I135

            Stokes = np.zeros((4, I[0].shape[0], I[0].shape[1]))

            Stokes[0] = I[0] + I[2]
            Stokes[1] = I[0] - I[2]
            Stokes[2] = I[1] - I[3]
            Stokes[3] = I[1] + I[3]


            val_temp = 0
            stokes_temp = 0
            s_zero = 0
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    I_temp = np.array([I[0, i, j], I[1, i, j], I[2, i, j], I[3, i, j]])
                    AS = np.array([0.5 * (Stokes[0, i, j] + Stokes[1, i, j]), 0.5 * (Stokes[0, i, j] + Stokes[2, i, j]),
                                   0.5 * (Stokes[0, i, j] - Stokes[1, i, j]), 0.5 * (Stokes[0, i, j] - Stokes[2, i, j])])
                    val_temp += np.linalg.norm(I_temp - AS) / (np.linalg.norm(I_temp) + np.linalg.norm(AS))
                    """if Stokes[0, i, j]**2 - Stokes[1, i, j]**2 - Stokes[2, i, j]**2 < 0:
                        stokes_temp += 1
                    if Stokes[0, i, j] < 0:
                        s_zero += 1"""
            i_as.append(val_temp / (image.shape[0]*image.shape[0]))
            stokes_param.append(stokes_temp / (image.shape[0] * image.shape[0]) * 100)
            stokes_zero.append(s_zero / (image.shape[0] * image.shape[0]) * 100)

        x = np.asarray(i_as)
        x = x[~np.isnan(x)]
        #y = np.asarray(stokes_param)
        #z = np.asarray(stokes_zero)

        print("Moyenne norm(I - AS) pour " + f + " : ", np.mean(x))
        print("Médiane norm(I - AS) pour " + f + " : ", np.median(x))
        print("Variance norm(I - AS) pour " + f + " : ", np.var(x))

        """print("Moyenne pourcentage S0² < S1² + S2² pour " + f + " : ", np.mean(y))
        print("Médiane npourcentage S0² < S1² + S2² pour " + f + " : ", np.median(y))
        print("Variance pourcentage S0² < S1² + S2² pour " + f + " : ", np.var(y))

        print("Moyenne pourcentage S0 < 0 pour " + f + " : ", np.mean(z))
        print("Médiane npourcentage S0 < 0 pour " + f + " : ", np.median(z))
        print("Variance pourcentage S0 < 0 pour " + f + " : ", np.var(z))"""

def plot_histogram_stokes(path_folder):
    folder_list = sorted(os.listdir(path_folder))
    plt.figure()
    for f in folder_list:
        imgs_polar = sorted(os.listdir(path_folder + "/" + f))
        stokes_param = []
        for k in imgs_polar:
            image = imageio.imread(path_folder + "/" + f + "/" + k)
            I = np.zeros((4, image.shape[0], image.shape[1]))
            I[0] = image[:, :, 0]  # I0
            I[1] = image[:, :, 1]  # I45
            I[2] = image[:, :, 2]  # I90
            I[3] = image[:, :, 3]  # I135

            Stokes = np.zeros((4, I[0].shape[0], I[0].shape[1]))

            Stokes[0] = I[0] + I[2]
            Stokes[1] = I[0] - I[2]
            Stokes[2] = I[1] - I[3]
            Stokes[3] = I[1] + I[3]

            stokes_temp = 0
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if Stokes[0, i, j] - Stokes[1, i, j] - Stokes[2, i, j] < 0:
                        stokes_temp += 1
            stokes_param.append(stokes_temp / (image.shape[0]*image.shape[0]))

        y = np.asarray(stokes_param)
        plt.plot(y)

    plt.title("Percentage of S0 < S1 + S2 for each image")
    plt.legend(["polar KITTI constraints", "polar KITTI no constraints", "real polar"])
    plt.savefig("/home/rblin/Documents/c2_kitti_polar_constraints.png")


#path_folder = "/home/rblin/Documents/Databases/kitti_polar_final/train/images"
#path_folder = "/home/rblin/Documents/Databases/kitti_polar_final/val/images"

#path_folder = "/home/rblin/Documents/Databases/examples_polar_kitti/RGB_to_polar"
#path_destination = "/home/rblin/Documents/Databases/examples_polar_kitti/RGB_to_polar_intensities"

#path_folder = "/home/rblin/Documents/Databases/kitti_polar_without_constraints/train/images"
#path_destination = "/home/rblin/Documents/Databases/kitti_polar_without_constraints_3ch/train/images"

#get_three_channels(path_folder, path_destination)

#get_four_intensities(path_folder, path_destination)

#path_folder = "/home/rblin/Documents/Databases/images_for_optical evaluation"

#plot_histogram_ias(path_folder)

path_folder = "/home/rblin/Documents/New_illustrations_ACCV/Polar_gen"
path_destination = "/home/rblin/Documents/New_illustrations_ACCV/Intensities"

get_four_intensities(path_folder, path_destination)




