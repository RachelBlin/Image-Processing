import os
import imageio
import cv2

path_folder = "AOP_moyen/ensoleille_fam"

files = os.listdir(path_folder)

for i in range(len(files)):
    image_temp = imageio.imread(path_folder + "/" + files[i])
    im_filtree = cv2.medianBlur(image_temp, 3)
    imageio.imwrite('AOP_filtre_median/ensoleille_fam/' + files[i], im_filtree)