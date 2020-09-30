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

def remove_night_images(path_labels):
    """
    A function to remove all the badly processed images

    :param path_labels: The path of the folder containing the images that need to be sorted
    """
    labels = sorted(os.listdir(path_labels))
    for k in range(len(labels)):
        content = open(path_labels + labels[k]).read()
        if 'night' in content:
            os.remove(path_labels + labels[k])

def remove_images_without_label(path_folder):
    """
    A function to remove images without labels

    :param path_folder: The path containing all the images
    :return:
    """


    #labels = os.listdir(path_folder + "labels/val/")
    labels = os.listdir(path_folder + "labels/val/")
    images = os.listdir(path_folder + "images/val/")
    for i in images:
        name_i = i.split(".")
        if name_i[0] + '.xml' not in labels:
            os.remove(path_folder + "images/val/" + i)

def remove_labels_without_images(path_folder):
    """
    A function to remove labels without images

    :param path_folder: The path containing all the images
    :return:
    """

    labels = os.listdir(path_folder + "LABELS_polar")
    images = os.listdir(path_folder + "POLAR")
    for l in labels:
        name_l = l.split(".")
        if name_l[0] + '.tiff' not in images:
            os.remove(path_folder + "LABELS_polar/" + l)

def rename_labels(path_folder):
    labels = os.listdir(path_folder + 'LABELS_polar/')
    for l in labels:
        name = l.split('.')
        number = name[0].split('_')
        print(name, number)
        os.rename(path_folder + 'LABELS_polar/' + l, path_folder + 'LABELS_polar/' + number[0] + '.' + name[1])

path_labels = '/home/rblin/Documents/Databases/BDD100K/BDD100K_polarNoC/labels/val/'
remove_night_images(path_labels)
path_folder = '/home/rblin/Documents/Databases/BDD100K/BDD100K_polarNoC/'
remove_images_without_label(path_folder)
