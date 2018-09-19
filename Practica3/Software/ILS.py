# -*- coding: utf-8 -*-
"""
Created on Mon May 22 22:46:28 2018

@author: adrianprodri
"""

from sklearn.metrics import accuracy_score  # Necesario para imprimir la precision de las predicciones
import numpy as np
from Evaluation import *


class ILS:
    def __init__(self, knn,length, X_train, y_train, min=0.0, max=1.0, alpha=0.5, variance=0.4, upper_bound=1.0, lower_bound=0.2, max_iterations=15, max_evaluations_BL=1000):
        self.knn = knn
        self.X_train = X_train
        self.y_train = y_train
        self.min = min
        self.max = max
        self.length = length
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_iterations = max_iterations
        self.t = round(0.1 * length)
        self.max_evaluations_BL = max_evaluations_BL
        self.alpha = alpha
        self.variance = variance

    def __generateSolution(self):
        generated = list(np.random.uniform(self.min, self.max, self.length))

        for n, i in enumerate(generated):
            if i > self.upper_bound:
                generated[n] = self.upper_bound
            elif i < self.lower_bound:
                generated[n] = 0.0

        return generated

    def __mutationOperator(self, actual):
        mut = np.random.normal(0, self.variance, 1)  # Gaussiana
        indexes = list(range(0,
                             self.length))  # Pesos que vamos a modificar para intentar alcanzar una solución más prometedora
        np.random.shuffle(indexes)

        for i in range(0, self.t):  # Modificamos t caracteristicas aleatorias
            actual[indexes[i]] = actual[indexes[i]] + mut

            if actual[indexes[i]] > self.upper_bound:
                actual[indexes[i]] = self.upper_bound
            elif actual[indexes[i]] < self.lower_bound:
                actual[indexes[i]] = 0.0

        return actual

    def __localSearch(self, actual):
        evaluations = 0

        while evaluations < self.max_evaluations_BL:
            indexes = list(range(0,
                                 self.length))  # Pesos que vamos a modificar para intentar alcanzar una solución más prometedora
            np.random.shuffle(indexes)
            original_evaluation = self.__fitnessCalculation(actual)
            evaluations += 1        # Por cada evaluacion de la BL
            new_solution = actual[:]

            new_solution = self.__mutationOperator(new_solution)

            new_evaluation = self.__fitnessCalculation(new_solution)
            evaluations += 1        # Por cada evaluacion de la BL

            if new_evaluation > original_evaluation:  # Se ha encontrado un mejor vecino por lo que salimos y volvemos a comenzar la BL sobre este nuevo vecino
                actual = new_solution[:]

        return actual

    def __fitnessCalculation(self, check):
        self.knn.fit(self.X_train * check, self.y_train)  # Ajustamos el modelo con los datos datos

        pred = self.knn.predict(self.X_train * check)  # Asi comprobamos la transformacion realizada a los datos, se respeta el leave one out
        prediction = accuracy_score(self.y_train, pred)

        return evaluacion(self.alpha, prediction, check.count(0.0), self.length)

    def start(self):
        actual_solution = self.__generateSolution()
        actual_solution = self.__localSearch(actual_solution)
        best_solution = actual_solution
        iteration = 1

        while iteration < self.max_iterations:
            iteration += 1
            if self.__fitnessCalculation(actual_solution) > self.__fitnessCalculation(best_solution):
                actual_solution = self.__mutationOperator(actual_solution)
                actual_solution = self.__localSearch(actual_solution)
            else:
                actual_solution = best_solution[:]
                actual_solution = self.__mutationOperator(actual_solution)
                actual_solution = self.__localSearch(actual_solution)

            if self.__fitnessCalculation(actual_solution) > self.__fitnessCalculation(best_solution):
                best_solution = actual_solution[:]

        return best_solution
