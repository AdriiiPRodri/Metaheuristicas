# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 18:25:55 2018

@author: adr_i
"""

# loading libraries
from scipy.io.arff import loadarff
import numpy as np
from sklearn.model_selection import StratifiedKFold
import math
import operator
from time import time
import warnings
warnings.filterwarnings('ignore')   # Cuando acabe colocarlo para evitar warinings de deprecated


################ KNN implementation ##########################

##### Similitud #####
def euclideanDistance(instance1, instance2, length):    # Comprueba componente a componente la distancia entre dos instancias de los dataset pasados
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)
#####################
    
##### getNeighbors #####
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1 # Funciona igual que c++ los indices
    for x in range(len(trainingSet)):
        if (testInstance != trainingSet[x]).any():
            dist = euclideanDistance(testInstance, trainingSet[x], length)
            distances.append((x, dist)) # A cada muestra del training le asigna una distancia
            distances.sort(key=operator.itemgetter(1)) # Ordena de menor distancia a mayor distancia, itemgetter(n) n es la dimensión
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
    return (100*(correct/float(len(testSet))))

#^^^^^^^^^^^^^^^^ KNN implementation ^^^^^^^^^^^^^^^^^^^#
    

################ RELIEF implementation ##########################    
##### getFriends ##### -> Devolverá de cada set de ejemplos que se le pasen sus k amigos más cercanos
def getFriends(train_atributos, train_etiquetas, ejemplo_atributos, ejemplo_etiquetas, k):
    distances = []
    length = len(ejemplo_atributos) - 1
    for x in range(len(train_etiquetas)):
        if (ejemplo_atributos != train_atributos[x]).any():
            dist = euclideanDistance(ejemplo_atributos, train_atributos[x], length)
            distances.append((x, dist)) # A cada muestra del training le asigna una distancia
            distances.sort(key=operator.itemgetter(1)) # Ordena de menor distancia a mayor distancia
    
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
            dist = euclideanDistance(ejemplo_atributos, train_atributos[x], length)
            distances.append((x, dist)) # A cada muestra del training le asigna una distancia
            distances.sort(key=operator.itemgetter(1)) # Ordena de menor distancia a mayor distancia
    
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
    return( 100 * (ceros / caracteristicas) )

def evaluacion(a, aciertos, instancias, ceros, caracteristicas):
    return( a * getAccuracy(instancias, aciertos) + (1 - a) * tasa_red(ceros, caracteristicas) )

#^^^^^^^^^^^^^^^^ Funciones ^^^^^^^^^^^^^^^^^^^#

# Comprobaciones

#tasa_class = 100 * precision
#tasa_red = 
        
############### Apertura y devision del dataset ###################
    
# Datasets disponibles
park = 'Instancias APC/parkinsons.arff'
ozone = 'Instancias APC/ozone-320.arff'
heart = 'Instancias APC/spectf-heart.arff'
iris = 'Instancias APC/iris.arff'
diabetes = 'Instancias APC/diabetes.arff'  # Tiempo de ejecución elevado
lungC = 'Instancias APC/lung-cancer.arff'  # Dataset muy pequeño, signo ? cambiado por 1 y 2 (dominio de ese atributo) de manera aleatoria por no poder tratarlos actualmente
breastC = 'Instancias APC/breast-cancer.arff'  # Falla al ejecutarse, contiene strings en los atributos

usado = ozone

# loading training data
with open(usado,'r') as f:
    data, meta = loadarff(f)

# create design matrix X and target vector y
X = data[meta.names()[:-1]]  #everything but the last column
X = X.view(np.float).reshape(data.shape + (-1,)) #converts the record array to a normal numpy array
y = data[meta.names()[-1]]
#y = y[:, np.newaxis]

# Normalizamos los atributos para no dar prioridad a unos sobre otros
for i in range(0,X.shape[1]):
    maximo = X.max(axis=0)[i]
    minimo = X.min(axis=0)[i]
    for j in range(0,X.shape[0]):
        X[j][i] = (X[j][i] - minimo) / (maximo - minimo)

# k-fold cross validation
fold = 5
a = 0.5     # Alfa de la función de evaluación
# Por defecto k = 1
k = 1 # Numero de vecinos que usamos para nuestro KNN, la implementación de todo la práctica funciona con cualquier n en [1, X_train.shape[0] - 1] siempre cogiendo un 
# número impar pues los empates tienen un compartimiento incierto y el límite superior de vecinos sería la muestra completa menos el propio que se examina de ahí el -1

kf = StratifiedKFold(n_splits = fold)

print("*******************************************************************")
print("KNN SIN PESOS")
i = 0
tiempo_medio = 0
media = 0
media_f = 0

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index] 
    prediccion = []
    tiempo_inicial = time()

    for x in range(len(X_test)):
        neighbors = getNeighbors(X_train, X_test[x], k)
        prediccion.append(getResponse(neighbors, y_train))
        
    media = media + getAccuracy(y_test, prediccion)
    media_f = media_f + evaluacion(a, prediccion, y_test, 0.0, X_train.shape[1])
    i += 1
    tiempo_final = time()
    tiempo_medio = tiempo_medio + (tiempo_final - tiempo_inicial)
    print("\n\t\tPartición", i, "SIMPLE acierto =", getAccuracy(y_test, prediccion), "%")
    print("\t\tTasa de reducción =", 0.0, "%")
    print("\t\tFunción de evaluación = ", evaluacion(a, prediccion, y_test, 0.0, X_train.shape[1]), "%")
    print("\t\tTiempo =", tiempo_final - tiempo_inicial)
    
print("\n\tDataset analizado:", usado)
print("\tMetodo de evaluación usado:", fold, "fold cross validation")
print("\tSIMPLE media acierto =",media/fold,"% ACIERTO")
print("\tTiempo medio =", tiempo_medio/fold)
print("\tTasa de reduccion media = 0.0")
print("\tFunción de evaluación media =", media_f/fold)
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
    w = relief(X_train, y_train, k, 0.2)
    X_train = X_train * w
    X_test = X_test * w

    for x in range(len(X_test)):
        neighbors_relief = getNeighbors(X_train, X_test[x], k)
        prediccion_relief.append(getResponse(neighbors_relief, y_train))
        
    media = media + getAccuracy(y_test, prediccion_relief)
    media_r = media_r + tasa_red(w.count(0.0), X_train.shape[1])
    media_f = media_f + evaluacion(a, prediccion_relief, y_test, w.count(0.0), X_train.shape[1])
    i += 1
    tiempo_final = time()
    tiempo_medio = tiempo_medio + (tiempo_final - tiempo_inicial)
    print("\n\t\tPartición", i, "RELIEF acierto =", getAccuracy(y_test, prediccion_relief))
    print("\t\tTasa de reducción =", tasa_red(w.count(0.0), X_train.shape[1]))
    print("\t\tFunción de evaluación = ", evaluacion(a, prediccion_relief, y_test, w.count(0.0), X_train.shape[1]), "%")
    print("\t\tTiempo =", tiempo_final - tiempo_inicial)
    
print("\n\tDataset analizado:", usado)
print("\tMetodo de evaluación usado:", fold, "fold cross validation")
print("\tRELIEF media acierto =",media/fold,"% ACIERTO")
print("\tTiempo medio =", tiempo_medio/fold)
print("\tTasa de reduccion media =", media_r/fold)
print("\tFunción de evaluación media =", media_f/fold)
print("\n")
print("*******************************************************************")
print("\n")
#################

print("*******************************************************************")
print("KNN CON PESOS BL")
# Colocamos la semilla para los aleatorios
np.random.seed(1) # Siempre los mismos aleatorios iniciales

vecinos_generados_tope = 150
varianza = 0.3
n_características = X_train.shape[1]
tope = 20 * n_características
i = 0
media = 0
media_r = 0
tiempo_medio = 0
media_f = 0

for train_index, test_index in kf.split(X, y):
    w = list(np.random.uniform(0.0, 1.0, X_train.shape[1]))  # Valores de menos de 0.2 se consideran caracterísitca no influyente
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    vecinos_generados = 0
    generadas = 0
    tiempo_inicial = time()
    media_e = 0
    
    while vecinos_generados < vecinos_generados_tope and generadas < tope:
        zi = np.random.normal(0, varianza, 1)       # Gaussiana
        indices_pesos = list(range(0,X_train.shape[1]))    # Pesos que vamos a modificar para intentar alcanzar una solución más prometedora
        np.random.shuffle(indices_pesos)
        
        for j in indices_pesos:
            prediccion = []
            w_anterior = w[:]        # En caso de no ser prometedora volvemos a la solución anterior y modificamos otro peso
            Wmax = max(w)
            w[j] = float(w[j] + zi)      # Aqui modificamos un peso con un zi que es aleatorio calculado anteriormente
            if Wmax != max(w):
                for u in range(len(w)):
                    if w[u] < 0.2:
                        w[u] = 0
                    else:
                        w[u] = w[u] / Wmax
            
            X_train_BL = X_train * w
                                                                                
            for x in range(len(X_train)):    # Comprobamos si nuestro nuevo vector de pesos nos proporciona una mejor solución que la anterior
                neighbors = getNeighbors(X_train_BL, X_train_BL[x], k)
                prediccion.append(getResponse(neighbors, y_train))
                                
            vecinos_generados += 1
            eva = evaluacion(a, prediccion, y_train, w.count(0.0), X_train.shape[1])

            if eva > media_e:
                media_e = eva
                generadas = 0
                break
            else:
                w = w_anterior[:]
                generadas += 1
                                        
    
    prediccion = []
    X_train_BL = X_train * w
    X_test_BL = X_test * w
    # Comprobamos para los pesos que hemos obtenido en BL:
    for x in range(len(X_test_BL)):    
        neighbors = getNeighbors(X_train_BL, X_test_BL[x], k)
        prediccion.append(getResponse(neighbors, y_train))
        
    accuracy = getAccuracy(y_test, prediccion)
    red = tasa_red(w.count(0.0), X_train.shape[1])
    eva = evaluacion(a, prediccion, y_test, w.count(0.0), X_train.shape[1])
    media = media + accuracy
    media_r = media_r + red
    media_f = media_f + eva
    i += 1
    tiempo_final = time()
    tiempo_medio = tiempo_medio + (tiempo_final - tiempo_inicial)
            
    print("\n\t\tBL partición =", i, "BL acierto =", accuracy," ACIERTO")
    print("\t\tTasa de reducción =", red)
    print("\t\tFunción de evaluación = ", eva, "%")
    print("\t\tTiempo =", tiempo_final - tiempo_inicial)




print("\n\tDataset analizado:", usado)
print("\tMetodo de evaluación usado:", fold, "fold cross validation")
print("\tBL media acierto", media/fold,"% ACIERTO")
print("\tTiempo medio =", tiempo_medio/fold)
print("\tTasa de reduccion media =", media_r/fold)
print("\tFunción de evaluación media =", media_f/fold)
print("\n")
print("*******************************************************************")
print("\n")
#################

    




####################### FROM SCIKIT-LEARN #######################################

## loading library
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import accuracy_score # Necesario para imprimir la precision de las predicciones
#
#X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 5)
#
## instantiate learning model (k = 3)
#knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
#
## fitting the model
#knn.fit(X_train, y_train)
#
## predict the response
#pred = knn.predict(X_test)
#
## evaluate accuracy
#print (accuracy_score(y_test, pred))