import os
from shutil import copyfile, move

def get_frames(path_folder):
    """
    A function to get a list of all the pictures in a folder

    :param path_folder: The path of the folder to get all the pictures from
    :return A list containing all the files in a folder
    """
    try:
        files = sorted(os.listdir(path_folder))
    except FileNotFoudError:
        print("No such file or directory ", path_folder)
    return files

def create_directory(path_directory):
    """
    A function to create a directory

    :param path_directory: The path of the directory to be created
    """
    try:
        os.mkdir(path_directory)
    except FileExistsError:
        print("Directory ", path_directory, " already exists")

def get_final_frames(path_folder, path_directory, first_frame, last_frame, step):
    """
    A function to get all the interesting polarimetric frames from a sequence of frames in order to make a database

    :param path_folder: The path of the folder containing all the frames
    :param path_directory: The path of the directory to put the final frames in
    :param first_frame: The index of the first frame
    :param last_frame: The index of the last frame
    :param step: The step between two frames we want to take into account
    :return: A list of the new frames
    """
    frames = get_frames(path_folder)
    create_directory(path_directory)
    new_frames = []
    for i in range(first_frame, last_frame, step):
        copyfile(path_folder + "/" + frames[i], path_directory + "/" + frames[i])
        new_frames.append(frames[i])
    return new_frames

def get_rgb_frames(path_folder, path_process):
    """
    A function to get all the interesting RGB frames from a sequence of frames in order to make a database

    :param path_folder: The path of the folder containing all the frames to be processed
    :param path_process: The path of the directory to put the final frames in
    """
    imgs_rgb = sorted(os.listdir(path_folder))
    for k in range(0, len(imgs_rgb), 50*4):
        copyfile(path_folder + imgs_rgb[k], path_process + imgs_rgb[k])

def move_files_to_right_folder(path_folder):
    """
    A function to sort the files in a folder

    :param path_folder: The path of the folder to be sorted
    """
    files = get_frames(path_folder)
    create_directory(path_folder + "/IMAGES")
    create_directory(path_folder + "/LABELS")
    for i in range(len(files)):
        if files[i].split(".")[1] == "png":
            move(path_folder + "/" + files[i], path_folder + "/IMAGES/" + files[i])
        elif files[i].split(".")[1] == "xml":
            move(path_folder + "/" + files[i], path_folder + "/LABELS/" + files[i])

def make_txt_voc_file(path_folder, file_name):
    """
    A function to make the txt file to train a network in the Pascal VOC format

    :param path_folder: The path of the folder containing the images and labels to be processed
    :param file_name: The name of the txt file
    """
    files = os.listdir(path_folder + "RGB_rs") # + 'I04590')
    file = path_folder + file_name
    f = open(file, 'w')
    for k in files:
        name = k.split('.')
        f.write(name[0] + '\n')
    f.close()

"""path_folder = "/home/rblin/Documents/Databases/Cerema/GoPro/brouillard_frames"
path_directory = "/home/rblin/Documents/Databases/POLARIMETRIC_DB_V2/18_09/jour/brouillard_20m/RGB"
first_frame = 375
last_frame = 3542
step = 60

get_final_frames(path_folder, path_directory, first_frame, last_frame, step)"""

#move_files_to_right_folder(path_folder)


path_folder = "/home/rblin/Documents/Databases/Final_DB/DB_POLAR_RGB_ITS/val_rgb/"
file_name = "val_rgb.txt"

make_txt_voc_file(path_folder, file_name)
