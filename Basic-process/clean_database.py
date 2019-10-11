import os
from shutil import copyfile

def copy_database(path_images, path_labels, path_final_images):
    """
    A function to keep only the labeled images

    :param path_images: The path of the folder containing all the images
    :param path_labels: The path of the folder containing all the labels
    :param path_final_images: The path to the folder where to put the copy of all the images
    """

    try:
        labels = sorted(os.listdir(path_labels))
    except FileNotFoudError:
        print("No such file or directory ", path_labels)

    try:
        images = sorted(os.listdir(path_images)) #+ "RetinaNet_I04590/"))
    except FileNotFoudError:
        print("No such file or directory ", path_images)

    """if not os.path.exists(path_final_images + "I04590/"):
        os.mkdir(path_final_images + "I04590/")

    if not os.path.exists(path_final_images + "I045135/"):
        os.mkdir(path_final_images + "I045135/")

    if not os.path.exists(path_final_images + "I090135/"):
        os.mkdir(path_final_images + "I090135/")

    if not os.path.exists(path_final_images + "I4590135/"):
        os.mkdir(path_final_images + "I4590135/")

    if not os.path.exists(path_final_images + "Params/"):
        os.mkdir(path_final_images + "Params/")

    if not os.path.exists(path_final_images + "Pauli2/"):
        os.mkdir(path_final_images + "Pauli2/")

    if not os.path.exists(path_final_images + "Pauli3/"):
        os.mkdir(path_final_images + "Pauli3/")

    if not os.path.exists(path_final_images + "Stokes/"):
        os.mkdir(path_final_images + "Stokes/")

    if not os.path.exists(path_final_images + "Rachel/"):
        os.mkdir(path_final_images + "Rachel/")

    if not os.path.exists(path_final_images + "Rachel2/"):
        os.mkdir(path_final_images + "Rachel2/")"""

    for k in range(len(images)):
        if str(k) + ".xml" in labels:
            copyfile(path_images + "/" + images[k],
                     path_final_images + "/" + images[k])
            """copyfile(path_images + "RetinaNet_I04590/" + str(k) + ".png",
                     path_final_images + "I04590/" + str(k) + ".png")
            copyfile(path_images + "RetinaNet_I045135/" + str(k) + ".png",
                     path_final_images + "I045135/" + str(k) + ".png")
            copyfile(path_images + "RetinaNet_I090135/" + str(k) + ".png",
                     path_final_images + "I090135/" + str(k) + ".png")
            copyfile(path_images + "RetinaNet_I4590135/" + str(k) + ".png",
                     path_final_images + "I4590135/" + str(k) + ".png")
            copyfile(path_images + "RetinaNet_Params/" + str(k) + ".png",
                     path_final_images + "Params/" + str(k) + ".png")
            copyfile(path_images + "RetinaNet_Pauli2/" + str(k) + ".png",
                     path_final_images + "Pauli2/" + str(k) + ".png")
            copyfile(path_images + "RetinaNet_Pauli3/" + str(k) + ".png",
                     path_final_images + "Pauli3/" + str(k) + ".png")
            copyfile(path_images + "RetinaNet_Stokes/" + str(k) + ".png",
                     path_final_images + "Stokes/" + str(k) + ".png")
            copyfile(path_images + "RetinaNet_Rachel/" + str(k) + ".png",
                     path_final_images + "Rachel/" + str(k) + ".png")
            copyfile(path_images + "RetinaNet_Rachel2/" + str(k) + ".png",
                     path_final_images + "Rachel2/" + str(k) + ".png")
            copyfile(path_labels + str(k) + ".xml",
                     path_final_labels + str(k) + ".xml")"""
        print(k)

def remove_bad_images(path_images):
    """
    A function to remove all the badly processed images

    :param path_images: The path of the folder containing the bad images
    """
    images = sorted(os.listdir(path_images))
    for k in range(len(images)):
        os.remove(path_images + images[k])

path_images = "/home/rblin/Documents/Databases/POLARIMETRIC_DB_V2/20_02/train_polar/POLAR"
path_labels = "/home/rblin/Documents/Databases/POLARIMETRIC_DB_V2/20_02/train_polar/LABELS"
path_final_images = "/home/rblin/Documents/Databases/POLARIMETRIC_DB_V2/20_02/train_polar/PROCESS"
#path_final_labels = "/home/rblin/Documents/New_polarimetric_DB_3239/train_polar/LABELS/"

copy_database(path_images, path_labels, path_final_images)

#remove_bad_images(path_images)