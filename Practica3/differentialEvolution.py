# -*- coding: utf-8 -*-
"""
Created on Mon May 24 12:46:28 2018

@author: adrianprodri
"""

import warnings

from operator import itemgetter

import numpy as np
from sklearn.metrics import accuracy_score  # Necesario para imprimir la precision de las predicciones

from Evaluation import *

warnings.filterwarnings('ignore')  # Cuando acabe colocarlo para evitar warnings de deprecated


class differentialEvolution:
    def __init__(self, knn, length, X_train, y_train, size=50, min=0.0, max=1.0, alpha=0.5, variance=0.4, F=0.5, cross_chance=0.5, upper_bound=1.0, lower_bound=0.2, max_evaluations=15000):
        self.knn = knn
        self.length = length
        self.size = size
        self.cross_chance = cross_chance
        self.min = min
        self.max = max
        self.alpha = alpha
        self.variance = variance
        self.F = F
        self.max_evaluations = max_evaluations
        self.actual_evaluation = 0
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.X_train = X_train
        self.y_train = y_train
        self.actual_generation = []

    def __individual(self):
        generated = list(np.random.uniform(self.min, self.max, self.length))

        for n, i in enumerate(generated):   # Contador y valor
            if i > self.upper_bound:
                generated[n] = self.upper_bound
            elif i < self.lower_bound:
                generated[n] = 0.0

        return generated

    def __generatePopulation(self):
        actual = []
        for i in range(0, self.size):
            actual.append(self.__individual())

        return actual

    def __fitnessCalculation(self, check):
        self.knn.fit(self.X_train * check, self.y_train)  # Ajustamos el modelo con los datos

        pred = self.knn.predict(self.X_train * check)
        prediction = accuracy_score(self.y_train, pred)

        self.actual_evaluation += 1

        return evaluacion(self.alpha, prediction, check.count(0.0), self.length)

    def __fatherSelection(self):
        return np.random.uniform(0, self.length, 3)     # Numero de padres

    def __bestOfGeneration(self, actual):
        ordened = [(self.__fitnessCalculation(i), i) for i in
                   actual]  # Calcula el fitness de cada individuo, y lo guarda en pares ordenados siendo la primera componente la valoración y la segunda individuo
        ordened = [i[1] for i in sorted(ordened, reverse=True, key=itemgetter(
            0))]  # Ordena los pares ordenados (de más fitness a menos fitness) y nos quedamos con los individuos

        return actual.index(ordened[0])

    def startRand(self):
        actual_population = self.__generatePopulation()

        while self.actual_evaluation < self.max_evaluations:
            for j in range(0, self.size):
                candidates = list(range(0, self.size))
                if j in candidates:
                    candidates.remove(j)  # EL mismo se excluye
                random_index = np.random.choice(candidates, 3, False)
                father1 = actual_population[random_index[0]]
                father2 = actual_population[random_index[1]]
                father3 = actual_population[random_index[2]]
                target = actual_population[j]
                offspring = []

                for k in range(0, self.length):
                    random_bound = np.random.uniform()
                    if random_bound < self.cross_chance:
                        diff = father1[k] + self.F * (father2[k] - father3[k])
                        if diff > self.upper_bound:
                            diff = self.upper_bound
                        elif diff < self.lower_bound:
                            diff = 0.0
                        offspring.append(diff)
                    else:
                        offspring.append(target[k])

                offspring_fitness = self.__fitnessCalculation(offspring)
                actual_population_fitness = self.__fitnessCalculation(target)

                if offspring_fitness > actual_population_fitness:
                    actual_population[j] = offspring[:]

        return actual_population[self.__bestOfGeneration(actual_population)]

    def startBest(self):
        actual_population = self.__generatePopulation()

        while self.actual_evaluation < self.max_evaluations:
            for j in range(0, self.size):
                candidates = list(range(0, self.size))
                if j in candidates:
                    candidates.remove(j)  # EL mismo se excluye
                random_index = np.random.choice(candidates, 3, False)
                father1 = actual_population[random_index[0]]
                father2 = actual_population[random_index[1]]
                actual_best = actual_population[self.__bestOfGeneration(actual_population)]
                target = actual_population[j]
                offspring = []

                for k in range(0, self.length):
                    random_bound = np.random.uniform()
                    if random_bound < self.cross_chance:
                        diff = target[k] + self.F * (actual_best[k] - target[k]) + self.F * (father1[k] - father2[k])
                        if diff > self.upper_bound:
                            diff = self.upper_bound
                        elif diff < self.lower_bound:
                            diff = 0.0
                        offspring.append(diff)
                    else:
                        offspring.append(target[k])

                offspring_fitness = self.__fitnessCalculation(offspring)
                actual_population_fitness = self.__fitnessCalculation(target)

                if offspring_fitness > actual_population_fitness:
                    actual_population[j] = offspring[:]

        return actual_population[self.__bestOfGeneration(actual_population)]
