import math
import operator

##### Similitud #####
def manhattanDistance(instance1, instance2,
                      length):  # Comprueba componente a componente la distancia entre dos instancias de los dataset pasados
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

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