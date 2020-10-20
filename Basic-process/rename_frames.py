import os
import shutil

def rename_frames_in_order(path_folder):
    """
    A function to rename raw polarimetric images in order to put them into the right order

    :param path_folder: The path of the folder containing the images to be renamed
    """
    try:
        files = sorted(os.listdir(path_folder))
    except FileNotFoudError:
        print("No such file or directory ", path_folder)

    for f in files:
        name = f.split("_")
        if len(name[4])==9:
            name[4] = '000' + name[4]
        elif len(name[3])==10:
            name[4] = '00' + name[4]
        elif len(name[3])==11:
            name[4] = '0' + name[4]
        f_new = name[4]
        print(f_new)
        os.rename(path_folder + f, path_folder + f_new)

def naive_rename_frames(path_folder):
    """
    A function to rename raw polarimetric images to respect the first set of the database

    :param path_folder: The path of the folder containing the images to be renamed
    """

    try:
        files = sorted(os.listdir(path_folder))
    except FileNotFoudError:
        print("No such file or directory ", path_folder)

    for k in range(len(files)):
        os.rename(path_folder + files[k], path_folder + str(k) + '.tiff')

def rename_frames_in_order_param(path_folder):
    """
    A function to rename the processed polarimetric images to put them into the right order

    :param path_folder: The folder containing the frames to be renamed
    """

    try:
        files = sorted(os.listdir(path_folder))
    except FileNotFoudError:
        print("No such file or directory ", path_folder)

    for f in files:
        name = f.split("_")
        if len(name[0])==1:
            name[0] = '000000' + name[0]
        elif len(name[0])==2:
            name[0] = '00000' + name[0]
        elif len(name[0])==3:
            name[0] = '0000' + name[0]
        elif len(name[0]) == 4:
            name[0] = '000' + name[0]
        elif len(name[0]) == 5:
            name[0] = '00' + name[0]
        elif len(name[0])==6:
            name[0] = '0' + name[0]
        f_new = name[0]
        os.rename(path_folder + f, path_folder + f_new + "_" + name[1])

def rename_rgb(path_rgb_movie):
    """
    A function to rename RGB images

    :param path_rgb_movie: The path of the folder containing all the images to be renamed
    """

    #sequences = sorted(os.listdir(path_rgb_movie))
    #len_seq = 0
    #for seq in sequences:
    files = os.listdir(path_rgb_movie) # + seq)
    if len(files) >=1:
        for f in files:
            name = f.split("frame")
            frame_number = name[1].split(".")
            nb = str(int(frame_number[0])) # + len_seq)
            if len(nb) == 1:
                nb = '000000' + nb
            elif len(nb) == 2:
                nb = '00000' + nb
            elif len(nb) == 3:
                nb = '0000' + nb
            elif len(nb) == 4:
                nb = '000' + nb
            elif len(nb) == 5:
                nb = '00' + nb
            elif len(nb) == 6:
                nb = '0' + nb
            os.rename(path_rgb_movie + "/" + f, path_rgb_movie + "/" + nb + ".png")
    #len_seq += len(files)

def rename_test_polar(path_test_polar):
    """
    A function to rename test_polar images

    :param path_rgb_movie: The path of the folder containing all the images to be renamed
    """

    files = os.listdir(path_test_polar)
    if len(files) >=1:
        for f in files:
            frame_number = f.split(".")
            nb = frame_number[0]
            if len(nb) == 1:
                nb = '000' + nb
            elif len(nb) == 2:
                nb = '00' + nb
            elif len(nb) == 3:
                nb = '0' + nb
            os.rename(path_test_polar + "/" + f, path_test_polar + "/" + nb + ".png")

def rename_frame_and_labels_fusion(path_polar, path_labels):
    """
    A function to rename polarimetric images so they're in the same order than their equivalent in RGB

    :param path_polar: Path to the polarimetric images folder
    """
    files = sorted(os.listdir(path_polar))
    labels = sorted(os.listdir(path_labels))
    if len(files) >= 1:
        for f in files:
            name = f.split(".")
            if name[0][-1] == "1" and int(name[0]) > 51510:
                frame_number = int(name[0]) - 34960
                if frame_number >= 0 and frame_number < 10:
                    nb = '000000' + str(frame_number)
                elif frame_number >= 10 and frame_number < 100:
                    nb = '00000' + str(frame_number)
                elif frame_number >= 100 and frame_number < 1000:
                    nb = '0000' + str(frame_number)
                elif frame_number >= 1000 and frame_number < 10000:
                    nb = '000' + str(frame_number)
                elif frame_number >= 10000 and frame_number < 100000:
                    nb = '00' + str(frame_number)
                elif frame_number >= 100000 and frame_number < 1000000:
                    nb = '0' + str(frame_number)
                if os.path.exists(os.path.join(path_polar, f)):
                    os.rename(os.path.join(path_polar, f), os.path.join(path_polar, nb + "_r.tiff"))
                    os.rename(os.path.join(path_labels, name[0] + '.xml'), os.path.join(path_labels, nb + "_r.xml"))
    files_rename = sorted(os.listdir(path_polar))
    for fr in files_rename:
        if '_r' in fr:
            new_name = fr.split('_')
            old_name = fr.split('.')
            os.rename(os.path.join(path_polar, fr), os.path.join(path_polar, new_name[0] + '.tiff'))
            os.rename(os.path.join(path_labels, old_name[0] + ".xml"), os.path.join(path_labels, new_name[0] + ".xml"))

def move_rgb(path_rgb_movie, final_path):
    """
    A function to move RGB images to another folder

    :param path_rgb_movie: The path from the RGB sequence
    :param final_path: The path where to move each sequence
    """
    sequences = sorted(os.listdir(path_rgb_movie))
    for seq in sequences:
        files = os.listdir(path_rgb_movie + seq)
        for f in files:
            shutil.move(path_rgb_movie + seq + "/" + f, final_path + f)


"""path_5 = "/media/rblin/87c4f13b-ad62-44ef-babf-70c3e7c8a343/polar/13_05_2019_15h_I/"

path_rgb_movie = "/home/rblin/Documents/Databases/11_05_2019/frames/"
final_path = "/home/rblin/Documents/Databases/11_05_2019/rgb/"
rename_rgb(path_rgb_movie)

move_rgb(path_rgb_movie, final_path)"""

#path_folder = "/home/rblin/Téléchargements/brouillard5/"

#rename_frames_in_order(path_folder)

#path_rgb_movie = "/home/rblin/Documents/Databases/Cerema/GoPro/brouillard_frames/"

#rename_rgb(path_rgb_movie)

#path_folder = "/home/rblin/Documents/Databases/20_02_POLAR/"

#path_polar = "/home/rblin/Documents/Databases/Final_DB/DB_POLAR_RGB_ITS/train_polar/POLAR"
#path_labels = "/home/rblin/Documents/Databases/Final_DB/DB_POLAR_RGB_ITS/train_polar/LABELS_polar"

#rename_frame_and_labels_fusion(path_polar, path_labels)

path_test_polar = "/home/rblin/Documents/Databases/PolarLITIS/test_polar/PARAM_POLAR/CosSin"

rename_test_polar(path_test_polar)