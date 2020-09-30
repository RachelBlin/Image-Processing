import os
import imageio
import numpy as np

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

def resize_image(path_image, path_new_image):
    """
    A function to resize an image

    :param path_image: The path of the image to be resized
    :param path_new_image: The path where to save the resized image
    """
    img_resize = np.zeros((557, 557, 3), dtype=int)
    image = imageio.imread(path_image)
    #print(image.shape)
    img_resize[557-480:557,:,:] = image[:, 112:669, :].copy()
    #img_resize = image[:, 112:669, :].copy()
    imageio.imwrite(path_new_image, img_resize)


def resize_images(path_images, path_new_images):
    """
    A function to resize a batch of images

    :param path_images: The path of the folder containing all the images to be resized
    :param path_new_images: The path of the folder where to put all the resized images
    """
    imgs = sorted(os.listdir(path_images))
    for img in imgs:
        name = img.split(".")
        resize_image(path_images + img, path_new_images + img)

def modify_labels(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    for object in root.findall('object'):
        bbox = object.find('bndbox')
        xmin = bbox.find('xmin')
        ymin = bbox.find('ymin')
        xmax = bbox.find('xmax')
        ymax = bbox.find('ymax')

        new_ymin = int(ymin.text) + 77
        ymin.text = str(new_ymin)
        new_ymax = int(ymax.text) + 77
        ymax.text = str(new_ymax)
        if int(xmin.text) >= 669 or int(xmax.text) <= 112:
            root.remove(object)
        elif int(xmin.text) <= 112 and int(xmax.text) > 112:
            new_xmin = 1
            xmin.text = str(new_xmin)
            new_xmax = int(xmax.text) - 112
            xmax.text = str(new_xmax)
        elif int(xmin.text) <= 669 and int(xmax.text) > 669:
            new_xmin = int(xmin.text) - 112
            xmin.text = str(new_xmin)
            new_xmax = 557
            xmax.text = str(new_xmax)
        else:
            new_xmin = int(xmin.text) - 112
            xmin.text = str(new_xmin)
            new_xmax = int(xmax.text) - 112
            xmax.text = str(new_xmax)

    tree.write(file_path)

def modify_xml_files(path_folder):
    files = os.listdir(path_folder)
    for f in files:
        modify_labels(os.path.join(path_folder, f))

#path_images = "/home/rblin/Documents/Databases/Final_DB/DB_POLAR_RGB_ITS/test_rgb/RGB/"
#path_new_images = "/home/rblin/Documents/Databases/Final_DB/DB_POLAR_RGB_ITS/test_rgb/RGB_rs/"

#resize_images(path_images, path_new_images)

path_folder = "/home/rblin/Documents/Databases/PolarLITIS/test_rgb/LABELS_RGB_rs/LABELS_RGB"
modify_xml_files(path_folder)
