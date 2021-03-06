import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
import math

path_folder = "DOP_filtre_median/brouillard_dm"

files = os.listdir(path_folder)

x_axis = [i for i in range(len(files))]
y_axis = []

n = len(files)

# Calcul de la liste de la moyenne des DOP par image
for i in range(n):
    image_temp = imageio.imread(path_folder + "/" + files[i])
    mean = np.mean(image_temp)
    y_axis.append(int(mean))

print(y_axis)

# Tri des valeurs
y = sorted(y_axis)
print(y)

# Calcul de la moyenne des valeurs
mean = 0
for c in y_axis:
    mean = mean + c

mean = mean / n

print("moyenne : ", mean)

# Calcul de la variance et de l'écart type
sd = 0
for j in y_axis:
    sd = sd + pow(j-mean, 2)

S_2 = sd
E_t = sd / n
sd = math.sqrt(E_t)

print("Somme des écarts à la moyenne : ", S_2)
print("Ecart-type : ", E_t)
print("Variance : ", sd)

# Calcul du coefficient d'assymétrie
CA = 0
for l in y_axis:
    CA = CA + pow((l-mean)/sd, 3)

CA = n/((n-1)*(n-2))*CA

print("Coefficeint d'assymétrie : ", CA)

# Calcul du coefficient d'applatissement
CAp = 0
for m in y_axis:
    CAp = CAp + pow((m-mean)/sd, 4)

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
#alpha = [0.4366, 0.3018, 0.2522, 0.2152, 0.1848, 0.1584, 0.1346, 0.1128, 0.0923, 0.0728, 0.0540, 0.0358, 0.0178, 0] # n=27
alpha = [0.4254, 0.2944, 0.2487, 0.2148, 0.1870, 0.1630, 0.1415, 0.1219, 0.1036, 0.0862, 0.0697, 0.0537, 0.0381, 0.0227, 0.0076] # n=30
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
#W_005 = 0.923 # n=27
W_005 = 0.927 # n=30

if W<W_005:
    print("Les données ne suivent pas la loi normale")
else :
    print("On ne peut pas dire que les données ne suivent pas une loi normale")

plt.figure(1)
plt.axhline(y=np.mean(y_axis), color='r', linestyle='-')
plt.plot(x_axis, y_axis, "b+")
plt.title("DOP moyen brouillard début de matinée")
plt.savefig("graphiques/DOP/filtre_median/brouillard_dm.png")
plt.show()

plt.figure(2)
plt.hist(y_axis, int(len(files)/5))
plt.savefig("graphiques/DOP/filtre_median/hist_brouillard_dm.png")
plt.show()
