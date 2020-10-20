import os
import imageio
import numpy as np
import cv2 as cv

def rgb_to_hsv(path_rgb, path_hsv):
    imgs = os.listdir(path_rgb)
    for im in imgs:
        rgb = imageio.imread(os.path.join(path_rgb, im))
        hsv = cv.cvtColor(rgb, cv.COLOR_RGB2HSV)
        if not os.path.exists(path_hsv):
            os.mkdir(path_hsv)
        imageio.imwrite(os.path.join(path_hsv, im.split(".")[0] + ".png"), hsv)

        """hsv = imageio.imread(os.path.join(path_hsv, im))
        rgb_back = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
        imageio.imwrite(os.path.join("/home/rblin/Documents/Databases/PolarLITIS/test_rgb/RS/HSV2", im.split(".")[0] + ".png"), rgb_back)"""

def rgb_to_lab(path_rgb, path_lab):
    imgs = os.listdir(path_rgb)
    for im in imgs:
        rgb = imageio.imread(os.path.join(path_rgb, im))
        lab = cv.cvtColor(rgb, cv.COLOR_RGB2Lab)
        if not os.path.exists(path_lab):
            os.mkdir(path_lab)
        imageio.imwrite(os.path.join(path_lab, im.split(".")[0] + ".png"), lab)

def rgb_to_YCrCb(path_rgb, path_ycrcb):
    imgs = os.listdir(path_rgb)
    for im in imgs:
        rgb = imageio.imread(os.path.join(path_rgb, im))
        ycrcb = cv.cvtColor(rgb, cv.COLOR_RGB2YCrCb)
        if not os.path.exists(path_ycrcb):
            os.mkdir(path_ycrcb)
        imageio.imwrite(os.path.join(path_ycrcb, im.split(".")[0] + ".png"), ycrcb)

path_rgb = "/home/rblin/Documents/Databases/PolarLITIS/test_rgb/RS/RGB"
path_hsv = "/home/rblin/Documents/Databases/PolarLITIS/test_rgb/RS/HSV"
path_lab = "/home/rblin/Documents/Databases/PolarLITIS/test_rgb/RS/LAB"
path_ycrcb = "/home/rblin/Documents/Databases/PolarLITIS/test_rgb/RS/YCrCb"

rgb_to_hsv(path_rgb, path_hsv)
rgb_to_lab(path_rgb, path_lab)
rgb_to_YCrCb(path_rgb, path_ycrcb)