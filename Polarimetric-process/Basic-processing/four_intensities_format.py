import os
import imageio
import numpy as np

def create_four_intensities(path_intensities, save_path):

    files_intensities = os.listdir(path_intensities)
    for f in files_intensities:
        img_intensities = imageio.imread(os.path.join(path_intensities,f))
        img_four_intensities = np.zeros((img_intensities.shape[0], img_intensities.shape[1], 4))
        img_four_intensities[:, :, :3] = img_intensities
        img_four_intensities[:, :, 3] = img_four_intensities[:, :, 0] + img_four_intensities[:, :, 2] - img_intensities[:, :, 1]
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        imageio.imwrite(os.path.join(save_path, f.split(".")[0] + ".png"), img_four_intensities)

path_intensities = "/home/rblin/Documents/Databases/PolarLITIS/test_polar/PARAM_POLAR/I04590"
save_path = "/home/rblin/Documents/Databases/PolarLITIS/test_polar/PARAM_POLAR/I04590135"

create_four_intensities(path_intensities, save_path)