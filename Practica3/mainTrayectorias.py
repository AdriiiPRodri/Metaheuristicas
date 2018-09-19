# -*- coding: utf-8 -*-
"""
Created on Mon May 22 22:46:28 2018

@author: adrianprodri
"""

import warnings

from time import time

from scipy.io.arff import loadarff
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import accuracy_score

from Evaluation import *
from Greedy import *
from simulatedAnnealing import *
from ILS import *
from differentialEvolution import *

warnings.filterwarnings('ignore')  # Cuando acabe colocarlo para evitar warnings de deprecated

############### Apertura y devision del dataset ###################

# Datasets disponibles
park = 'Instancias APC/parkinsons.arff'
ozone = 'Instancias APC/ozone-320.arff'
heart = 'Instancias APC/spectf-heart.arff'
diabetes = 'Instancias APC/diabetes.arff'  # Tiempo de ejecución elevado
sonar = 'Instancias APC/Sonar.arff'
wdbc = 'Instancias APC/Wdbc.arff'
spambase = 'Instancias APC/Spambase.arff'

usado = park

# loading training data
with open(usado, 'r') as f:
    data, meta = loadarff(f)

# create design matrix X and target vector y
X_d = data[meta.names()[:-1]]  # everything but the last column
X_d = X_d.view(np.float).reshape(data.shape + (-1,))  # converts the record array to a normal numpy array
y = data[meta.names()[-1]]
y_s = y

# Eliminamos filas duplicadas:
indices = []
contador = 0
seen = set()
X = []
for item in X_d:
    t = tuple(item)
    if t not in seen:
        X.append(item)
        seen.add(t)
    else:
        indices.append(contador)

    contador += 1

indices = np.array(indices)
contador = 0

while (contador != len(indices)):
    y = np.delete(y, (indices[contador]), axis=0)
    contador += 1
    indices -= 1

X = np.asarray(X)

del X_d, t, indices
##############################

# Normalizamos los atributos para no dar prioridad a unos sobre otros
for i in range(0, X.shape[1]):
    maximo = X.max(axis=0)[i]
    minimo = X.min(axis=0)[i]
    for j in range(0, X.shape[0]):
        X[j][i] = (X[j][i] - minimo) / (maximo - minimo)

# k-fold cross validation
fold = 5
a = 0.5  # Alfa de la función de evaluación
# Por defecto k = 1
k = 1  # Numero de vecinos que usamos para nuestro KNN, la implementación de todo la práctica funciona con cualquier n en [1, X_train.shape[0] - 1] siempre cogiendo un
# número impar pues los empates tienen un compartimiento incierto y el límite superior de vecinos sería la muestra completa menos el propio que se examina de ahí el -1

kf = StratifiedKFold(n_splits=fold, shuffle=False)
# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=k, p=1,
                           metric='manhattan')  # Uso la distancia Manhattan debido a que computacionalmente es muy poco costosa y no usa sqrt

print("\n###################################################################")
print("#################### ESQUEMA DE REPRESENTACION ####################")
print("###################################################################")

print("\n\tLos datos apareceran siguiendo el siguiente esquema:")
print("\t\t0º.- Semilla usada")
print("\t\t1º.- Tasa de clasificacion")
print("\t\t2º.- Tasa de reduccion")
print("\t\t3º.- Agregado")
print("\t\t4º.- Tiempo de ejecucion")

print("\n\tDespues de la ejecucion de todas las particiones apareceran las medias en este orden:")
print("\t\t1º.- Media de clasificacion")
print("\t\t2º.- Media de reduccion")
print("\t\t3º.- Media de agregado")
print("\t\t4º.- Media de tiempo de ejecucion")

print("\n###################################################################")
print("###################################################################")
print("###################################################################")

print("\nKNN USADO =>", k, "\n")
print("\nDataset analizado:", usado, "\n")

print("*******************************************************************")
print("KNN CON PESOS A 1")
i = 0
tiempo_medio = 0
media = 0
media_f = 0
limite_considerar = 0.2

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    prediccion = []
    tiempo_inicial = time()

    # fitting the model
    knn.fit(X_train, y_train)

    # predict the response
    pred = knn.predict(X_test)

    #    for x in range(len(X_test)):
    #        neighbors = getNeighbors(X_train, X_test[x], k)
    #        prediccion.append(getResponse(neighbors, y_train))
    #
    prediccion = accuracy_score(y_test, pred)
    media = media + prediccion
    media_f = media_f + evaluacion(a, prediccion, 0.0, X_train.shape[1])
    i += 1
    tiempo_final = time()
    tiempo_medio = tiempo_medio + (tiempo_final - tiempo_inicial)
    print("\n\n\t\t", i)
    print("\n\t\t", 100 * prediccion)
    print("\t\t 0.0")
    print("\t\t", evaluacion(a, prediccion, 0.0, X_train.shape[1]))
    print("\t\t", tiempo_final - tiempo_inicial)

print("\n\t", (media / fold) * 100)
print("\t 0.0")
print("\t", media_f / fold)
print("\t", tiempo_medio / fold)
print("\n")
print("*******************************************************************")
print("\n")
#################

print("*******************************************************************")
print("KNN CON PESOS RELIEF")
i = 0
tiempo_medio = 0
media = 0
media_r = 0
media_f = 0

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    prediccion_relief = []
    tiempo_inicial = time()
    w = relief(X_train, y_train, k, limite_considerar)
    X_train = X_train * w
    X_test = X_test * w

    # fitting the model
    knn.fit(X_train, y_train)

    # predict the response
    pred = knn.predict(X_test)

    prediccion = accuracy_score(y_test, pred)
    media = media + prediccion
    red = tasa_red(w.count(0.0), X_train.shape[1])
    media_r = media_r + red
    media_f = media_f + evaluacion(a, prediccion, 0.0, X_train.shape[1])
    i += 1
    tiempo_final = time()
    tiempo_medio = tiempo_medio + (tiempo_final - tiempo_inicial)
    print("\n\n\t\t", i)
    print("\n\t\t", 100 * prediccion)
    print("\t\t", red)
    print("\t\t", evaluacion(a, prediccion, w.count(0.0), X_train.shape[1]))
    print("\t\t", tiempo_final - tiempo_inicial)

print("\n\t", (media / fold) * 100)
print("\t", media_r / fold)
print("\t", media_f / fold)
print("\t", tiempo_medio / fold)
print("\n")
print("*******************************************************************")
print("\n")
#################

print("*******************************************************************")
print("KNN CON SIMULATED ANNEALING")

media = 0
media_r = 0
tiempo_medio = 0
media_f = 0
i = 0
length = X_train.shape[1]
mu = fi = 0.3

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    tiempo_inicial = time()
    media_e = 0
    np.random.seed(i + 1)
    simann = simulatedAnnealing(knn, mu, fi, length, X_train, y_train)

    w = simann.simannealing()

    prediccion = []

    # fitting the model
    knn.fit(X_train *  w, y_train)

    X_test = X_test * w

    # predict the response
    pred = knn.predict(X_test)

    prediccion = accuracy_score(y_test, pred)

    red = tasa_red(w.count(0.0), X_train.shape[1])
    eva = evaluacion(a, prediccion, w.count(0.0), X_train.shape[1])
    media = media + prediccion
    media_r = media_r + red
    media_f = media_f + eva
    i += 1
    tiempo_final = time()
    tiempo_medio = tiempo_medio + (tiempo_final - tiempo_inicial)

    print("\n\n\t\t", i)
    print("\n\t\t", 100 * prediccion)
    print("\t\t", red)
    print("\t\t", eva)
    print("\t\t", tiempo_final - tiempo_inicial)

print("\n\t", (media / fold) * 100)
print("\t", media_r / fold)
print("\t", media_f / fold)
print("\t", tiempo_medio / fold)
print("\n")
print("*******************************************************************")
print("\n")
#################

print("*******************************************************************")
print("KNN CON ILS")

media = 0
media_r = 0
tiempo_medio = 0
media_f = 0
i = 0
length = X_train.shape[1]

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    tiempo_inicial = time()
    media_e = 0
    np.random.seed(i + 1)
    ils = ILS(knn, length, X_train, y_train)

    w = ils.start()

    prediccion = []

    # fitting the model
    knn.fit(X_train * w, y_train)

    X_test = X_test * w

    # predict the response
    pred = knn.predict(X_test)

    prediccion = accuracy_score(y_test, pred)

    red = tasa_red(w.count(0.0), X_train.shape[1])
    eva = evaluacion(a, prediccion, w.count(0.0), X_train.shape[1])
    media = media + prediccion
    media_r = media_r + red
    media_f = media_f + eva
    i += 1
    tiempo_final = time()
    tiempo_medio = tiempo_medio + (tiempo_final - tiempo_inicial)

    print("\n\n\t\t", i)
    print("\n\t\t", 100 * prediccion)
    print("\t\t", red)
    print("\t\t", eva)
    print("\t\t", tiempo_final - tiempo_inicial)

print("\n\t", (media / fold) * 100)
print("\t", media_r / fold)
print("\t", media_f / fold)
print("\t", tiempo_medio / fold)
print("\n")
print("*******************************************************************")
print("\n")
#################

print("*******************************************************************")
print("KNN CON DIFFERENTIAL EVOLUTION Variante RAND")

media = 0
media_r = 0
tiempo_medio = 0
media_f = 0
i = 0
length = X_train.shape[1]

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    tiempo_inicial = time()
    media_e = 0
    np.random.seed(i + 1)
    diffEvo = differentialEvolution(knn, length, X_train, y_train)

    w = diffEvo.startRand()

    prediccion = []

    # fitting the model
    knn.fit(X_train * w, y_train)

    X_test = X_test * w

    # predict the response
    pred = knn.predict(X_test)

    prediccion = accuracy_score(y_test, pred)

    red = tasa_red(w.count(0.0), X_train.shape[1])
    eva = evaluacion(a, prediccion, w.count(0.0), X_train.shape[1])
    media = media + prediccion
    media_r = media_r + red
    media_f = media_f + eva
    i += 1
    tiempo_final = time()
    tiempo_medio = tiempo_medio + (tiempo_final - tiempo_inicial)

    print("\n\n\t\t", i)
    print("\n\t\t", 100 * prediccion)
    print("\t\t", red)
    print("\t\t", eva)
    print("\t\t", tiempo_final - tiempo_inicial)

print("\n\t", (media / fold) * 100)
print("\t", media_r / fold)
print("\t", media_f / fold)
print("\t", tiempo_medio / fold)
print("\n")
print("*******************************************************************")
print("\n")
#################

print("*******************************************************************")
print("KNN CON DIFFERENTIAL EVOLUTION Variante BEST")

media = 0
media_r = 0
tiempo_medio = 0
media_f = 0
i = 0
length = X_train.shape[1]

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    tiempo_inicial = time()
    media_e = 0
    np.random.seed(i + 1)
    diffEvo = differentialEvolution(knn, length, X_train, y_train)

    w = diffEvo.startBest()

    prediccion = []

    # fitting the model
    knn.fit(X_train * w, y_train)

    X_test = X_test * w

    # predict the response
    pred = knn.predict(X_test)

    prediccion = accuracy_score(y_test, pred)

    red = tasa_red(w.count(0.0), X_train.shape[1])
    eva = evaluacion(a, prediccion, w.count(0.0), X_train.shape[1])
    media = media + prediccion
    media_r = media_r + red
    media_f = media_f + eva
    i += 1
    tiempo_final = time()
    tiempo_medio = tiempo_medio + (tiempo_final - tiempo_inicial)

    print("\n\n\t\t", i)
    print("\n\t\t", 100 * prediccion)
    print("\t\t", red)
    print("\t\t", eva)
    print("\t\t", tiempo_final - tiempo_inicial)

print("\n\t", (media / fold) * 100)
print("\t", media_r / fold)
print("\t", media_f / fold)
print("\t", tiempo_medio / fold)
print("\n")
print("*******************************************************************")
print("\n")
#################