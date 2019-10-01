import os
import numpy as np
import imageio
import math
import random
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.ndimage.filters import gaussian_filter1d

#path_folder1 = "DOP_moyen/brouillard_dm"
path_folder2 = "DOP_moyen/brumeux_fam"
path_folder3 = "DOP_moyen/couvert_dm"
path_folder4 = "DOP_moyen/ensoleille_fam"

#files1 = os.listdir(path_folder1)
files2 = os.listdir(path_folder2)
files3 = os.listdir(path_folder3)
files4 = os.listdir(path_folder4)

files = files2 + files3 + files4

maximum = np.zeros((len(files),))

"""for i in range(len(files1)):
    image_temp = imageio.imread(path_folder1 + "/" + files1[i])
    unique, counts = np.unique(image_temp, return_counts=True)
    unique = unique/255 # seulement pour DOP
    max = np.argmax(counts[1:len(counts)])
    maximum[i] = unique[max+1]"""

for j in range(len(files2)):
    image_temp = imageio.imread(path_folder2 + "/" + files2[j])
    unique, counts = np.unique(image_temp, return_counts=True)
    unique = unique / 255  # seulement pour DOP
    max = np.argmax(counts[1:len(counts)])
    #maximum[len(files1) + j] = unique[max + 1]
    maximum[j] = unique[max + 1]

for k in range(len(files3)):
    image_temp = imageio.imread(path_folder3 + "/" + files3[k])
    unique, counts = np.unique(image_temp, return_counts=True)
    unique = unique / 255  # seulement pour DOP
    max = np.argmax(counts[1:len(counts)])
    #maximum[len(files1) + len(files2) + k] = unique[max + 1]
    maximum[len(files2) + k] = unique[max + 1]

for l in range(len(files4)):
    image_temp = imageio.imread(path_folder4 + "/" + files4[l])
    unique, counts = np.unique(image_temp, return_counts=True)
    unique = unique / 255  # seulement pour DOP
    max = np.argmax(counts[1:len(counts)])
    #maximum[len(files1) + len(files2) + len(files3) + l] = unique[max + 1]
    maximum[len(files2) + len(files3) + l] = unique[max + 1]

valeurs = random.sample(maximum.tolist(), 50)

n = len(valeurs)

print("Nombre d'éléments : ", n)

print("Eléments : ", valeurs)

# Tri des valeurs
y = sorted(valeurs)
print("Liste triée : ", y)

# Calcul de la moyenne des valeurs
med = 0
for c in valeurs:
    med = med + c

med = med / n

print("moyenne : ", med)

# Calcul de la variance et de l'écart type
sd = 0
for j in valeurs:
    sd = sd + pow(j-med, 2)

S_2 = sd
E_t = sd / n
sd = math.sqrt(E_t)

print("Somme des écarts à la moyenne : ", S_2)
print("Ecart-type : ", E_t)
print("Variance : ", sd)

# Calcul du coefficient d'assymétrie
CA = 0
for l in valeurs:
    CA = CA + pow((l-med)/sd, 3)

CA = n/((n-1)*(n-2))*CA

print("Coefficeint d'assymétrie : ", CA)

# Calcul du coefficient d'applatissement
CAp = 0
for m in valeurs:
    CAp = CAp + pow((m-med)/sd, 4)

CAp = n*(n+1)/((n-1)*(n-2)*(n-3))*CAp-3*pow(n-1,2)/((n-2)*(n-3))

print("Coefficient d'applatissement : ", CAp)

"""Test de Shapiro-Wilk"""

# Calcul de d
d = []
for a in range(n):
    d_temp = y[n-a-1] - y[a]
    d.append(d_temp)

# Calcul de k
if n%2==0:
    k = int(n/2)
else:
    k = int((n-1)/2)

# Calcul de b_2
# lien vers la table des alpha : http://www.biostat.ulg.ac.be/pages/Site_r/normalite_files/Table-alpha.pdf
#alpha = [0.4068, 0.2813, 0.2415, 0.2121, 0.1883, 0.1678, 0.1496, 0.1331, 0.1179, 0.1036, 0.0900, 0.0770, 0.0645, 0.0523, 0.0404, 0.0287, 0.0172, 0.0057] # n=36
#alpha = [0.4643, 0.3185, 0.2578, 0.2119, 0.1736, 0.1399, 0.1092, 0.0804, 0.0530, 0.0263, 0] # n=21
#alpha = [0.4096, 0.2834, 0.2427, 0.2127, 0.1883, 0.1673, 0.1487, 0.1317, 0.1160, 0.1013, 0.0873, 0.0739, 0.0610, 0.0484, 0.0361, 0.0239, 0.0119, 0] # n=35
#alpha = [0.4366, 0.3018, 0.2522, 0.2152, 0.1848, 0.1584, 0.1346, 0.1128, 0.0923, 0.0728, 0.0540, 0.0358, 0.0178, 0] # n=27
#alpha = [0.4254, 0.2944, 0.2487, 0.2148, 0.1870, 0.1630, 0.1415, 0.1219, 0.1036, 0.0862, 0.0697, 0.0537, 0.0381, 0.0227, 0.0076] # n=30
alpha = [0.3751, 0.2574, 0.2260, 0.2032, 0.1847, 0.1691, 0.1554, 0.1430, 0.1317, 0.1212, 0.1113, 0.1020, 0.0932, 0.0846, 0.0764, 0.0685, 0.0608, 0.0532, 0.0459, 0.0386, 0.0314, 0.0244, 0.0174, 0.0104, 0.0035] # n=50
b_2 = 0
for j in range(k):
    b_2 = b_2 + alpha[j]*d[j]

b_2 = pow(b_2, 2)

print("b carré : ", b_2)

# Calcul de W
# lien vers la table des valeurs de W0.05 : http://www.biostat.ulg.ac.be/pages/Site_r/normalite_files/table-W.png
W = b_2/S_2

print("W :", W)

#W_005 = 0.935 # n=36
#W_005 = 0.908 # n=21
#W_005 = 0.934 # n=35
#W_005 = 0.923 # n=27
#W_005 = 0.927 # n=30
W_005 = 0.947 # n=50

if W<W_005:
    print("Les données ne suivent pas la loi normale")
else :
    print("On ne peut considérer avec une confiance de 95% que les données suivent une loi normale")

x_axis= [i for i in range(len(files))]

plt.figure(1)
l1 = plt.axhline(y=np.mean(maximum), color='r', linestyle='-')
l2 = plt.axhline(y=np.median(maximum), color='c', linestyle='-')
l3 = plt.axhline(y=np.percentile(maximum, 75), color='y', linestyle='--')
plt.axhline(y=np.percentile(maximum, 25), color='y', linestyle='--')
plt.plot(x_axis, sorted(maximum), "b+")
y_smooth = gaussian_filter1d(sorted(maximum), sigma=2)
plt.plot(x_axis, y_smooth, "g")
plt.legend((l1, l2, l3), ('moyenne', 'mediane', '1er et 3e quartiles'))
plt.title("DOP max tous temps confondus")
plt.savefig("graphiques/DOP/classique/DOP_max_sb_sorted.png")
plt.show()

"""plt.figure(2)
m, bins, patches = plt.hist(maximum, 20, normed=1, facecolor='blue', alpha=2)
norm = mlab.normpdf(bins, np.mean(maximum), np.std(maximum))
ll = plt.plot(bins, norm, 'r--')
plt.savefig("graphiques/DOP/classique/hist_DOP_max.png")
plt.show()"""

"""plt.figure(3)
m, bins, patches = plt.hist(y, 20, normed=1, facecolor='blue', alpha=2)
norm = mlab.normpdf(bins, np.mean(y), sd)
ll = plt.plot(bins, norm, 'r--')
plt.savefig("graphiques/DOP/classique/hist_DOP_max_50_sb.png")
plt.show()"""