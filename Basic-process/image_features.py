import numpy as np

from matplotlib import image
from matplotlib import pyplot as plt

from skimage.filters.rank import entropy
from skimage.morphology import disk, square
from skimage.transform import resize

def local_probability(patch, pixel_value):
    """

    :param patch: The part of an image on which we want to compute the local probability of a pixel
    :param pixel_value: The value of the pixel on which we want to compute the local probability
    :return: the probability
    """
    p = 0
    M = patch.shape[0]
    N = patch.shape[1]

    for i in range(M):
        for j in range(N):
            p += dirac(patch[i, j] - pixel_value)

    p = p/(M*N)

    return p

def dirac(x):
    if x == 0:
        return 1
    else:
        return 0

"""arr = np.array([[1, 2], [1, 2]])
arr2 = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
print(local_probability(arr2, 3))"""

def compute_local_entropy(patch):
    pixels = np.unique(patch)
    pi = np.array([local_probability(patch, pix) for pix in pixels])
    entropy = pi * np.log2(pi)
    return sum(-entropy)

def compute_global_entropy(image, window_size):
    h = image.shape[0]
    w = image.shape[1]
    entropy = np.zeros((h,w))
    for index, _ in np.ndenumerate(image):
        x, y = index
        if x + window_size > h:
            upper_bound_x = h
        else:
            upper_bound_x = x + window_size
        if y + window_size > w:
            upper_bound_y = w
        else:
            upper_bound_y = y + window_size
        entropy[x,y] = compute_local_entropy(image[x:upper_bound_x, y:upper_bound_y])

    return entropy

#print(compute_local_entropy(arr2))

#img = image.imread('/home/rblin/Documents/New_illustrations_ACCV/Polar/0001611_I135.png')
#img = image.imread('/home/rblin/Documents/New_illustrations_ACCV/PARAM_POLAR/AOP/0001611_AOP.png')
#img = image.imread('/home/rblin/Documents/New_illustrations_ACCV/PARAM_POLAR/DOP/0001611_DOP.png')
#img = image.imread('/home/rblin/Documents/New_illustrations_ACCV/PARAM_POLAR/Stokes2/0001611_S2.png')
#img = image.imread('/home/rblin/Documents/Databases/Illustrations_ITS/PARAM_POLAR/I/0_I135.png')
#img = image.imread('/home/rblin/Documents/Databases/Illustrations_ITS/PARAM_POLAR/Params/0_DOP.png')
img = image.imread('/home/rblin/Documents/Databases/Illustrations_ITS/PARAM_POLAR/Stokes/0_S2.png')

ent = compute_global_entropy(np.round(img*255, decimals=0).astype(int), 4)

ent2 = entropy(img, square(4))

plt.imshow(ent, cmap='Greys')
plt.show()
plt.imshow(ent2, cmap='Greys')
plt.show()