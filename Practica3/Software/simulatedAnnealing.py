# -*- coding: utf-8 -*-
"""
Created on Mon May 22 22:46:28 2018

@author: adrianprodri
"""

from sklearn.metrics import accuracy_score  # Necesario para imprimir la precision de las predicciones
import numpy as np
from Evaluation import *


class simulatedAnnealing:
    def __init__(self, knn, mu, fi, length, X_train, y_train, min=0.0, max=1.0, alpha=0.5, final_temperature=0.001, upper_bound=1.0, lower_bound=0.2, max_evaluations=15000):
        self.final_temperature = final_temperature
        self.knn = knn
        self.X_train = X_train
        self.y_train = y_train
        self.min = min
        self.max = max
        self.length = length
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_neighbors = 10 * length
        self.max_success = length
        self.max_evaluations = max_evaluations
        self.actual_evaluation = 0
        self.alpha = alpha
        self.actual_solution = self.__generateSolution()
        self.actual_temperature = (mu * self.__fitnessCalculation(self.actual_solution)) / -np.log(fi)
        self.initial_temperature = self.actual_temperature

    def __generateSolution(self):
        generated = list(np.random.uniform(self.min, self.max, self.length))

        for n, i in enumerate(generated):
            if i > self.upper_bound:
                generated[n] = self.upper_bound
            elif i < self.lower_bound:
                generated[n] = 0.0

        return generated

    def __generateNeighbor(self, variance=0.3):
        index = np.random.randint(low=0, high=self.length - 1, size=1)
        mut = np.random.normal(0, variance, 1)  # Gaussiana
        neighbor = self.actual_solution[:]
        neighbor[index] = neighbor[index] + mut

        if neighbor[index] > self.upper_bound:
            neighbor[index] = self.upper_bound
        elif neighbor[index] < self.lower_bound:
            neighbor[index] = 0.0

        return neighbor

    def __fitnessCalculation(self, check):
        self.knn.fit(self.X_train * check, self.y_train)  # Ajustamos el modelo con los datos datos

        self.actual_evaluation += 1

        pred = self.knn.predict(self.X_train * check)  # Asi comprobamos la transformacion realizada a los datos, se respeta el leave one out
        prediction = accuracy_score(self.y_train, pred)

        return evaluacion(self.alpha, prediction, check.count(0.0), self.length)

    def __annealing(self, number_of_cooling):
        return (self.initial_temperature - self.final_temperature) / (
                    number_of_cooling * self.initial_temperature * self.final_temperature)

    def __finish(self):
        return self.actual_evaluation >= self.max_evaluations or self.actual_number_success == 0

    def simannealing(self):
        best_solution = self.actual_solution
        self.actual_number_success = 1  # Para poder entrar en el bucle principal, simplemente es para entrar luego lo colocaremos a 0 al principio del bucle
        iteration = 0

        while not self.__finish() and self.actual_temperature > self.final_temperature:  # No enfriaremos a menos de la temperatura final por esto puede que no lleguemos a las 15000 iteraciones (max_exitos), ademas tenemos las restricciones del pdf de la practica
            self.actual_number_success = 0
            iteration += 1

            for i in range(0, self.max_neighbors):
                neighbor = self.__generateNeighbor()
                eva_neighbor = self.__fitnessCalculation(neighbor)
                eva_actual = self.__fitnessCalculation(self.actual_solution)
                deferential = eva_actual - eva_neighbor  # s - s' (invertido con respecto pseudo codigo pues buscamos maximizar y de este modo conseguiremos que la formula del if siguiente se mantenga igual al pseudo codigo)
                random = np.random.uniform(0, 1)
                if deferential < 0.0 or random <= np.exp(-deferential / self.actual_temperature):
                    self.actual_solution = neighbor
                    self.actual_number_success += 1  # Consideramos exito este vecino pues ahora es nuestra solucion actual
                    if eva_neighbor > self.__fitnessCalculation(best_solution):     # Aqui nos ahorramos una evaluacion pues como hemos aceptado el vecino y teniamos su evaluacion ya hecha pues la usamos
                        best_solution = self.actual_solution
                    if self.actual_number_success >= self.max_success:  # Aqui tenemos la segun condicion del bucle L(t) la cual puede provocar que salgamos antes
                        break

            self.actual_temperature = self.__annealing(iteration)

        return best_solution