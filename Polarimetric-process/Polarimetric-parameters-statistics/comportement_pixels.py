import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import math

path_folder = "AOP_moyen/couvert_fm"

files = os.listdir(path_folder)

maximum = np.zeros((len(files),))
for i in range(len(files)):
    image_temp = imageio.imread(path_folder + "/" + files[i])
    unique, counts = np.unique(image_temp, return_counts=True)
    # unique = unique/255 # seulement pour DOP
    plt.figure(i+1)
    max = np.argmax(counts[1:len(counts)])
    plt.axvline(x=unique[max+1], color='r', linestyle='-')
    plt.plot(unique[1:len(unique)], counts[1:len(counts)], "b+")
    counts_smooth = gaussian_filter1d(counts[1:len(counts)], sigma=2)
    plt.plot(unique[1:len(unique)], counts_smooth, "g")
    plt.title("Nb pixels en fonction de la valeur de l'AOP")
    plt.savefig("graphiques/AOP/classique/couvert_fm/couvert_fm_moyen_" + files[i])
    maximum[i] = unique[max+1]

plt.figure()
line1 = plt.plot(maximum, "b+")
line2 = plt.axhline(y=np.mean(maximum), color='r', linestyle='-')
line3 = plt.axhline(y=np.median(maximum), color='g', linestyle='-')
plt.legend((line1, line2, line3), ("valeur la plus présente de l'AOP par image", 'moyenne', 'médiane'))
plt.title("Résumé pour l'AOP couvert fin de matinée")
plt.savefig("graphiques/AOP/classique/couvert_fm/couvert_moyen_resume.png")

# Test statistique sur les maximums de DOP

n = len(maximum)

print("Nombre d'éléments : ", n)

print("Eléments : ", maximum)

# Tri des valeurs
y = sorted(maximum)
print("Liste triée : ", y)

# Calcul de la moyenne des valeurs
med = 0
for c in maximum:
    med = med + c

med = med / n

print("moyenne : ", med)

# Calcul de la variance et de l'écart type
sd = 0
for j in maximum:
    sd = sd + pow(j-med, 2)

S_2 = sd
E_t = sd / n
sd = math.sqrt(E_t)

print("Somme des écarts à la moyenne : ", S_2)
print("Ecart-type : ", E_t)
print("Variance : ", sd)

# Calcul du coefficient d'assymétrie
CA = 0
for l in maximum:
    CA = CA + pow((l-med)/sd, 3)

CA = n/((n-1)*(n-2))*CA

print("Coefficeint d'assymétrie : ", CA)

# Calcul du coefficient d'applatissement
CAp = 0
for m in maximum:
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
alpha = [0.4096, 0.2834, 0.2427, 0.2127, 0.1883, 0.1673, 0.1487, 0.1317, 0.1160, 0.1013, 0.0873, 0.0739, 0.0610, 0.0484, 0.0361, 0.0239, 0.0119, 0] # n=35
#alpha = [0.4366, 0.3018, 0.2522, 0.2152, 0.1848, 0.1584, 0.1346, 0.1128, 0.0923, 0.0728, 0.0540, 0.0358, 0.0178, 0] # n=27
#alpha = [0.4254, 0.2944, 0.2487, 0.2148, 0.1870, 0.1630, 0.1415, 0.1219, 0.1036, 0.0862, 0.0697, 0.0537, 0.0381, 0.0227, 0.0076] # n=30
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
W_005 = 0.934 # n=35
#W_005 = 0.923 # n=27
#W_005 = 0.927 # n=30

if W<W_005:
    print("Les données ne suivent pas la loi normale")
else :
    print("On ne peut considérer avec une confiance de 95% que les données suivent une loi normale")