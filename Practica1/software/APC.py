# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 18:25:55 2018

@author: Adrián Jesús Peña Rodríguez
"""

# loading libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score  # Necesario para imprimir la precision de las predicciones
from scipy.io.arff import loadarff
import numpy as np
from sklearn.model_selection import StratifiedKFold
import math
import operator
from time import time
import warnings

warnings.filterwarnings('ignore')  # Cuando acabe colocarlo para evitar warnings de deprecated


################ KNN implementation ##########################

##### Similitud #####
def manhattanDistance(instance1, instance2,
                      length):  # Comprueba componente a componente la distancia entre dos instancias de los dataset pasados
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


#####################

##### getNeighbors ##### -> Sin uso, ineficiente, faltaría implementar kdtree para más velocidad. Usado scikit-learn en vez de esta función para encontrar a los vecinos
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1  # Funciona igual que c++ los indices
    for x in range(len(trainingSet)):
        if (testInstance != trainingSet[x]).any():
            dist = manhattanDistance(testInstance, trainingSet[x], length)
            distances.append((x, dist))  # A cada muestra del training le asigna una distancia
            distances.sort(key=operator.itemgetter(
                1))  # Ordena de menor distancia a mayor distancia, itemgetter(n) n es la dimensión
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


########################

##### Votación de etiquetas en los vecinos #####
def getResponse(neighbors, data):
    classVotes = {}
    for x in range(len(neighbors)):
        response = data[neighbors[x]]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


################################################

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet)))


# ^^^^^^^^^^^^^^^^ KNN implementation ^^^^^^^^^^^^^^^^^^^#


################ RELIEF implementation ##########################
##### getFriends ##### -> Devolverá de cada set de ejemplos que se le pasen sus k amigos más cercanos
def getFriends(train_atributos, train_etiquetas, ejemplo_atributos, ejemplo_etiquetas, k):
    distances = []
    length = len(ejemplo_atributos) - 1
    for x in range(len(train_etiquetas)):
        if (ejemplo_atributos != train_atributos[x]).any():
            dist = manhattanDistance(ejemplo_atributos, train_atributos[x], length)
            distances.append((x, dist))  # A cada muestra del training le asigna una distancia
            distances.sort(key=operator.itemgetter(1))  # Ordena de menor distancia a mayor distancia

    friends = []
    x = 0
    y = 0

    while y < k:
        if train_etiquetas[distances[y + x][0]] == ejemplo_etiquetas:
            friends.append(distances[y + x][0])
            y += 1
        else:
            x += 1
    return friends


##### getEnemies ##### -> Devolverá de cada set de ejemplos que se le pasen sus k enemigos más cercanos
def getEnemies(train_atributos, train_etiquetas, ejemplo_atributos, ejemplo_etiquetas, k):
    distances = []
    length = len(ejemplo_atributos) - 1
    for x in range(len(train_etiquetas)):
        if (ejemplo_atributos != train_atributos[x]).any():
            dist = manhattanDistance(ejemplo_atributos, train_atributos[x], length)
            distances.append((x, dist))  # A cada muestra del training le asigna una distancia
            distances.sort(key=operator.itemgetter(1))  # Ordena de menor distancia a mayor distancia

    enemies = []
    x = 0
    y = 0

    while y < k:
        if train_etiquetas[distances[y + x][0]] != ejemplo_etiquetas:
            enemies.append(distances[y + x][0])
            y += 1
        else:
            x += 1
    return enemies


def relief(datosTr, etiquetasTr, length, limite_considerar):
    w = [0] * datosTr.shape[1]

    for i in range(datosTr.shape[0]):
        ee = getEnemies(datosTr, etiquetasTr, datosTr[i], etiquetasTr[i], length)
        ea = getFriends(datosTr, etiquetasTr, datosTr[i], etiquetasTr[i], length)
        for j in range(datosTr.shape[1]):
            w[j] = w[j] + abs(datosTr[i][j] - datosTr[ee[0]][j]) - abs(datosTr[i][j] - datosTr[ea[0]][j])

    Wmax = max(w)

    for i in range(len(w)):
        if w[i] < limite_considerar:
            w[i] = 0
        else:
            w[i] = w[i] / Wmax

    return w


################ Funciones a evaluar ##########################

def tasa_red(ceros, caracteristicas):
    return (100 * (ceros / caracteristicas))


def evaluacion(a, prediccion, ceros, caracteristicas):
    return (a * prediccion * 100 + (1 - a) * tasa_red(ceros, caracteristicas))


# ^^^^^^^^^^^^^^^^ Funciones ^^^^^^^^^^^^^^^^^^^#

# Comprobaciones

# tasa_class = 100 * precision
# tasa_red =

############### Apertura y devision del dataset ###################

# Datasets disponibles
park = 'Instancias APC/parkinsons.arff'
ozone = 'Instancias APC/ozone-320.arff'
heart = 'Instancias APC/spectf-heart.arff'
iris = 'Instancias APC/iris.arff'
diabetes = 'Instancias APC/diabetes.arff'  # Tiempo de ejecución elevado
lungC = 'Instancias APC/lung-cancer.arff'  # Dataset muy pequeño, signo ? cambiado por 1 y 2 (dominio de ese atributo) de manera aleatoria por no poder tratarlos actualmente
breastC = 'Instancias APC/breast-cancer.arff'  # Falla al ejecutarse, contiene strings en los atributos

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
print("KNN CON PESOS BL")

vecinos_generados_tope = 15000
varianza = 0.3
n_características = X.shape[1]
tope = 20 * n_características
i = 0
media = 0
media_r = 0
tiempo_medio = 0
media_f = 0

for train_index, test_index in kf.split(X, y):
    # Colocamos la semilla para los aleatorios
    np.random.seed(i + 1)  # Siempre los mismos aleatorios iniciales
    w = list(
        np.random.uniform(0.0, 1.0, X.shape[1]))  # Valores de menos de 0.2 se consideran caracterísitca no influyente
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    vecinos_generados = 0
    generadas = 0
    tiempo_inicial = time()
    media_e = 0

    while vecinos_generados < vecinos_generados_tope and generadas < tope:
        zi = np.random.normal(0, varianza, 1)  # Gaussiana
        indices_pesos = list(
            range(0, X.shape[1]))  # Pesos que vamos a modificar para intentar alcanzar una solución más prometedora
        np.random.shuffle(indices_pesos)

        for j in indices_pesos:
            prediccion = []
            w_anterior = w[:]  # En caso de no ser prometedora volvemos a la solución anterior y modificamos otro peso
            Wmax = max(w)
            w[j] = float(w[j] + zi)  # Aqui modificamos un peso con un zi que es aleatorio calculado anteriormente
            if Wmax != max(w):
                for u in range(len(w)):
                    if w[u] < limite_considerar:
                        w[u] = 0
                    elif w[u] > 1:
                        w[u] = 1

            X_train_BL = X_train * w

            # fitting the model
            knn.fit(X_train * w, y_train)

            # predict the response
            pred = knn.predict(X_train_BL)

            prediccion = accuracy_score(y_train, pred)

            vecinos_generados += 1
            eva = evaluacion(a, prediccion, w.count(0.0), X_train.shape[1])

            if eva > media_e:
                media_e = eva
                generadas = 0
                break
            else:
                w = w_anterior[:]
                generadas += 1

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
