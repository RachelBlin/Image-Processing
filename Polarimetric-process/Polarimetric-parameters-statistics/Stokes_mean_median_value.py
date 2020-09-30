import os
import numpy as np
import imageio
import math
import random
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.ndimage.filters import gaussian_filter1d

path_folder = "/home/rblin/Documents/polar_for_statistics/generated_polar/car"

files = os.listdir(path_folder)

maximum = np.zeros((len(files), 3))
mean = np.zeros((len(files), 3))
median = np.zeros((len(files), 3))

for i in range(len(files)):
    image_temp = imageio.imread(path_folder + "/" + files[i])
    Stokes = np.zeros((3, image_temp.shape[0], image_temp.shape[1]))
    Stokes[0, :, :] = image_temp[:, :, 0]
    Stokes[1, :, :] = image_temp[:, :, 1]
    Stokes[2, :, :] = image_temp[:, :, 2]
    unique0, counts0 = np.unique(Stokes[0, :, :], return_counts=True)
    unique1, counts1 = np.unique(Stokes[1, :, :], return_counts=True)
    unique2, counts2 = np.unique(Stokes[2, :, :], return_counts=True)
    #unique = unique/255 # seulement pour DOP
    max0 = np.argmax(counts0[1:len(counts0)])
    max1 = np.argmax(counts1[1:len(counts1)])
    max2 = np.argmax(counts2[1:len(counts2)])
    mean0 = np.mean(Stokes[0, :, :])
    mean1 = np.mean(Stokes[1, :, :])
    mean2 = np.mean(Stokes[2, :, :])
    median0 = np.median(Stokes[0, :, :])
    median1 = np.median(Stokes[1, :, :])
    median2 = np.median(Stokes[2, :, :])
    maximum[i, 0] = unique0[max0 + 1]
    maximum[i, 1] = unique0[max1 + 1]
    maximum[i, 2] = unique0[max2 + 1]
    mean[i, 0] = mean0
    mean[i, 1] = mean1
    mean[i, 2] = mean2
    median[i, 0] = median0
    median[i, 1] = median1
    median[i, 2] = median2

save_path = "generated_polar/graph_car"

x_axis= [i for i in range(len(files))]

plt.figure(1)
l1 = plt.axhline(y=np.mean(maximum[:, 0]), color='r', linestyle='-')
l2 = plt.axhline(y=np.median(maximum[:, 0]), color='c', linestyle='-')
l3 = plt.axhline(y=np.percentile(maximum[:, 0], 75), color='y', linestyle='--')
plt.axhline(y=np.percentile(maximum[:, 0], 25), color='y', linestyle='--')
plt.plot(x_axis, sorted(maximum[:, 0]), "b+")
y_smooth = gaussian_filter1d(sorted(maximum[:, 0]), sigma=2)
plt.plot(x_axis, y_smooth, "g")
plt.legend((l1, l2, l3), ('moyenne', 'mediane', '1er et 3e quartiles'))
plt.title("répartition des S0 max")
plt.savefig("/home/rblin/Documents/polar_for_statistics/"+save_path+"/répartition_S0_max.png")
#plt.show()

plt.figure(2)
l1 = plt.axhline(y=np.mean(mean[:, 0]), color='r', linestyle='-')
l2 = plt.axhline(y=np.median(mean[:, 0]), color='c', linestyle='-')
l3 = plt.axhline(y=np.percentile(mean[:, 0], 75), color='y', linestyle='--')
plt.axhline(y=np.percentile(mean[:, 0], 25), color='y', linestyle='--')
plt.plot(x_axis, sorted(mean[:, 0]), "b+")
y_smooth = gaussian_filter1d(sorted(mean[:, 0]), sigma=2)
plt.plot(x_axis, y_smooth, "g")
plt.legend((l1, l2, l3), ('moyenne', 'mediane', '1er et 3e quartiles'))
plt.title("répartition des S0 mean")
plt.savefig("/home/rblin/Documents/polar_for_statistics/"+save_path+"/répartition_S0_mean.png")
#plt.show()

plt.figure(3)
l1 = plt.axhline(y=np.mean(median[:, 0]), color='r', linestyle='-')
l2 = plt.axhline(y=np.median(median[:, 0]), color='c', linestyle='-')
l3 = plt.axhline(y=np.percentile(median[:, 0], 75), color='y', linestyle='--')
plt.axhline(y=np.percentile(median[:, 0], 25), color='y', linestyle='--')
plt.plot(x_axis, sorted(median[:, 0]), "b+")
y_smooth = gaussian_filter1d(sorted(median[:, 0]), sigma=2)
plt.plot(x_axis, y_smooth, "g")
plt.legend((l1, l2, l3), ('moyenne', 'mediane', '1er et 3e quartiles'))
plt.title("répartition des S0 médian")
plt.savefig("/home/rblin/Documents/polar_for_statistics/"+save_path+"/répartition_S0_med.png")
#plt.show()

plt.figure(4)
m, bins, patches = plt.hist(maximum, 20, normed=1, facecolor='blue', alpha=2)
norm = mlab.normpdf(bins, np.mean(maximum[:, 0]), np.std(maximum[:, 0]))
ll = plt.plot(bins, norm, 'r--')
plt.title("répartition des S0 max")
plt.savefig("/home/rblin/Documents/polar_for_statistics/"+save_path+"/hist_S0_max.png")
#plt.show()

plt.figure(5)
m, bins, patches = plt.hist(mean, 20, normed=1, facecolor='blue', alpha=2)
norm = mlab.normpdf(bins, np.mean(mean[:, 0]), np.std(mean[:, 0]))
ll = plt.plot(bins, norm, 'r--')
plt.title("répartition des S0 moyen")
plt.savefig("/home/rblin/Documents/polar_for_statistics/"+save_path+"/hist_S0_med.png")
#plt.show()

plt.figure(6)
m, bins, patches = plt.hist(median, 20, normed=1, facecolor='blue', alpha=2)
norm = mlab.normpdf(bins, np.mean(median[:, 0]), np.std(median[:, 0]))
ll = plt.plot(bins, norm, 'r--')
plt.title("répartition des S0 médian")
plt.savefig("/home/rblin/Documents/polar_for_statistics/"+save_path+"/hist_S0_moy.png")
#plt.show()

plt.figure(7)
l1 = plt.axhline(y=np.mean(maximum[:, 1]), color='r', linestyle='-')
l2 = plt.axhline(y=np.median(maximum[:, 1]), color='c', linestyle='-')
l3 = plt.axhline(y=np.percentile(maximum[:, 1], 75), color='y', linestyle='--')
plt.axhline(y=np.percentile(maximum[:, 1], 25), color='y', linestyle='--')
plt.plot(x_axis, sorted(maximum[:, 1]), "b+")
y_smooth = gaussian_filter1d(sorted(maximum[:, 1]), sigma=2)
plt.plot(x_axis, y_smooth, "g")
plt.legend((l1, l2, l3), ('moyenne', 'mediane', '1er et 3e quartiles'))
plt.title("répartition des S1 max")
plt.savefig("/home/rblin/Documents/polar_for_statistics/"+save_path+"/répartition_S1_max.png")
#plt.show()

plt.figure(8)
l1 = plt.axhline(y=np.mean(mean[:, 1]), color='r', linestyle='-')
l2 = plt.axhline(y=np.median(mean[:, 1]), color='c', linestyle='-')
l3 = plt.axhline(y=np.percentile(mean[:, 1], 75), color='y', linestyle='--')
plt.axhline(y=np.percentile(mean[:, 1], 25), color='y', linestyle='--')
plt.plot(x_axis, sorted(mean[:, 1]), "b+")
y_smooth = gaussian_filter1d(sorted(mean[:, 1]), sigma=2)
plt.plot(x_axis, y_smooth, "g")
plt.legend((l1, l2, l3), ('moyenne', 'mediane', '1er et 3e quartiles'))
plt.title("répartition des S1 mean")
plt.savefig("/home/rblin/Documents/polar_for_statistics/"+save_path+"/répartition_S1_mean.png")
#plt.show()

plt.figure(9)
l1 = plt.axhline(y=np.mean(median[:, 1]), color='r', linestyle='-')
l2 = plt.axhline(y=np.median(median[:, 1]), color='c', linestyle='-')
l3 = plt.axhline(y=np.percentile(median[:, 1], 75), color='y', linestyle='--')
plt.axhline(y=np.percentile(median[:, 1], 25), color='y', linestyle='--')
plt.plot(x_axis, sorted(median[:, 1]), "b+")
y_smooth = gaussian_filter1d(sorted(median[:, 1]), sigma=2)
plt.plot(x_axis, y_smooth, "g")
plt.legend((l1, l2, l3), ('moyenne', 'mediane', '1er et 3e quartiles'))
plt.title("répartition des S1 médian")
plt.savefig("/home/rblin/Documents/polar_for_statistics/"+save_path+"/répartition_S1_med.png")
#plt.show()

plt.figure(10)
m, bins, patches = plt.hist(maximum, 20, normed=1, facecolor='blue', alpha=2)
norm = mlab.normpdf(bins, np.mean(maximum[:, 1]), np.std(maximum[:, 1]))
ll = plt.plot(bins, norm, 'r--')
plt.title("répartition des S1 max")
plt.savefig("/home/rblin/Documents/polar_for_statistics/"+save_path+"/hist_S1_max.png")
#plt.show()

plt.figure(11)
m, bins, patches = plt.hist(mean, 20, normed=1, facecolor='blue', alpha=2)
norm = mlab.normpdf(bins, np.mean(mean[:, 1]), np.std(mean[:, 1]))
ll = plt.plot(bins, norm, 'r--')
plt.title("répartition des S1 moyen")
plt.savefig("/home/rblin/Documents/polar_for_statistics/"+save_path+"/hist_S1_med.png")
#plt.show()

plt.figure(12)
m, bins, patches = plt.hist(median, 20, normed=1, facecolor='blue', alpha=2)
norm = mlab.normpdf(bins, np.mean(median[:, 1]), np.std(median[:, 1]))
ll = plt.plot(bins, norm, 'r--')
plt.title("répartition des S1 médian")
plt.savefig("/home/rblin/Documents/polar_for_statistics/"+save_path+"/hist_S1_moy.png")
#plt.show()

plt.figure(13)
l1 = plt.axhline(y=np.mean(maximum[:, 2]), color='r', linestyle='-')
l2 = plt.axhline(y=np.median(maximum[:, 2]), color='c', linestyle='-')
l3 = plt.axhline(y=np.percentile(maximum[:, 2], 75), color='y', linestyle='--')
plt.axhline(y=np.percentile(maximum[:, 2], 25), color='y', linestyle='--')
plt.plot(x_axis, sorted(maximum[:, 2]), "b+")
y_smooth = gaussian_filter1d(sorted(maximum[:, 2]), sigma=2)
plt.plot(x_axis, y_smooth, "g")
plt.legend((l1, l2, l3), ('moyenne', 'mediane', '1er et 3e quartiles'))
plt.title("répartition des S2 max")
plt.savefig("/home/rblin/Documents/polar_for_statistics/"+save_path+"/répartition_S2_max.png")
#plt.show()

plt.figure(14)
l1 = plt.axhline(y=np.mean(mean[:, 2]), color='r', linestyle='-')
l2 = plt.axhline(y=np.median(mean[:, 2]), color='c', linestyle='-')
l3 = plt.axhline(y=np.percentile(mean[:, 2], 75), color='y', linestyle='--')
plt.axhline(y=np.percentile(mean[:, 2], 25), color='y', linestyle='--')
plt.plot(x_axis, sorted(mean[:, 2]), "b+")
y_smooth = gaussian_filter1d(sorted(mean[:, 2]), sigma=2)
plt.plot(x_axis, y_smooth, "g")
plt.legend((l1, l2, l3), ('moyenne', 'mediane', '1er et 3e quartiles'))
plt.title("répartition des S2 mean")
plt.savefig("/home/rblin/Documents/polar_for_statistics/"+save_path+"/répartition_S2_mean.png")
#plt.show()

plt.figure(15)
l1 = plt.axhline(y=np.mean(median[:, 2]), color='r', linestyle='-')
l2 = plt.axhline(y=np.median(median[:, 2]), color='c', linestyle='-')
l3 = plt.axhline(y=np.percentile(median[:, 2], 75), color='y', linestyle='--')
plt.axhline(y=np.percentile(median[:, 2], 25), color='y', linestyle='--')
plt.plot(x_axis, sorted(median[:, 2]), "b+")
y_smooth = gaussian_filter1d(sorted(median[:, 2]), sigma=2)
plt.plot(x_axis, y_smooth, "g")
plt.legend((l1, l2, l3), ('moyenne', 'mediane', '1er et 3e quartiles'))
plt.title("répartition des S2 médian")
plt.savefig("/home/rblin/Documents/polar_for_statistics/"+save_path+"/répartition_S2_med.png")
#plt.show()

plt.figure(16)
m, bins, patches = plt.hist(maximum, 20, normed=1, facecolor='blue', alpha=2)
norm = mlab.normpdf(bins, np.mean(maximum[:, 2]), np.std(maximum[:, 2]))
ll = plt.plot(bins, norm, 'r--')
plt.title("répartition des S2 max")
plt.savefig("/home/rblin/Documents/polar_for_statistics/"+save_path+"/hist_S2_max.png")
#plt.show()

plt.figure(17)
m, bins, patches = plt.hist(mean, 20, normed=1, facecolor='blue', alpha=2)
norm = mlab.normpdf(bins, np.mean(mean[:, 2]), np.std(mean[:, 2]))
ll = plt.plot(bins, norm, 'r--')
plt.title("répartition des S2 moyen")
plt.savefig("/home/rblin/Documents/polar_for_statistics/"+save_path+"/hist_S2_med.png")
#plt.show()

plt.figure(18)
m, bins, patches = plt.hist(median, 20, normed=1, facecolor='blue', alpha=2)
norm = mlab.normpdf(bins, np.mean(median[:, 2]), np.std(median[:, 2]))
ll = plt.plot(bins, norm, 'r--')
plt.title("répartition des S2 médian")
plt.savefig("/home/rblin/Documents/polar_for_statistics/"+save_path+"/hist_S2_moy.png")
#plt.show()