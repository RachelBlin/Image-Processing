import os
import imageio
import numpy as np

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

path_images = "/home/rblin/Documents/Databases/Final_DB/DB_POLAR_RGB_ITS/test_rgb/RGB/"
path_new_images = "/home/rblin/Documents/Databases/Final_DB/DB_POLAR_RGB_ITS/test_rgb/RGB_rs/"

resize_images(path_images, path_new_images)