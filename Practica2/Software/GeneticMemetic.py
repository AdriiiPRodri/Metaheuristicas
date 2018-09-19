# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 12:46:28 2018

@author: adrianprodri
"""

import warnings

from operator import itemgetter

import numpy as np
from sklearn.metrics import accuracy_score  # Necesario para imprimir la precision de las predicciones

from Evaluation import *

warnings.filterwarnings('ignore')  # Cuando acabe colocarlo para evitar warnings de deprecated


class GeneticMemetic:
    def __init__(self, knn, length, size, iterations, X_train, y_train, cross_chance=1, mut_chance=0.001, minimun=0.0,
                 maximun=1.0, upper_bound=1.0, lower_bound=0.2):
        self.knn = knn
        self.length = length
        self.size = size
        self.cross_chance = cross_chance
        self.mut_chance = mut_chance
        self.min = minimun
        self.max = maximun
        self.actual_generation = []  # Lista de individuos actuales
        self.partial_new_generation = []  # Candidatos
        self.pressure = round(self.cross_chance * (self.size / 2))
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.actual_fitness = 0
        self.X_train = X_train
        self.y_train = y_train
        self.iterations = iterations + 1  # Para las divisiones
        self.actual_iteration = 1
        self.enterLS = False
        self.iteLS = 0.0
        self.number_mutation = round(self.mut_chance * (self.length * self.size))
        self.__generatePopulation()

    def __individual(self):
        generated = list(np.random.uniform(self.min, self.max, self.length))

        for n, i in enumerate(generated):
            if i > self.upper_bound:
                generated[n] = self.upper_bound
            elif i < self.lower_bound:
                generated[n] = 0.0

        return generated

    def __generatePopulation(self):
        for i in range(0, self.size):
            self.actual_generation.append(self.__individual())

    def __fitnessCalculation(self, check, alpha):
        self.knn.fit(self.X_train * check, self.y_train)  # Ajustamos el modelo con los datos

        pred = self.knn.predict(self.X_train * check)
        prediction = accuracy_score(self.y_train, pred)

        self.actual_iteration += 1

        if self.iteLS != 0.0 and self.actual_iteration % self.iteLS == 0.0:
            self.enterLS = True

        return evaluacion(alpha, prediction, check.count(0.0), self.length)

    def __binaryTournament(self, a=0.5):
        randoms = np.random.randint(low=0, high=self.size, size=2)
        parentA = self.__fitnessCalculation(self.partial_new_generation[randoms[0]], a)
        parentB = self.__fitnessCalculation(self.partial_new_generation[randoms[1]], a)

        if parentA >= parentB:
            return randoms[0]
        else:
            return randoms[1]

    def __crossOperatorBLX(self, selection, a):
        parent_union = self.actual_generation[selection[0]] + self.actual_generation[selection[1]]
        Cmax = max(parent_union)
        Cmin = min(parent_union)
        I = Cmax - Cmin
        childA = list(np.random.uniform(Cmin - I * a, Cmax + I * a, self.length))
        childB = list(np.random.uniform(Cmin - I * a, Cmax + I * a, self.length))

        for i in range(0, self.length):
            if childA[i] > self.upper_bound:
                childA[i] = self.upper_bound
            elif childA[i] < self.lower_bound:
                childA[i] = 0.0

            if childB[i] > self.upper_bound:
                childB[i] = self.upper_bound
            elif childB[i] < self.lower_bound:
                childB[i] = 0.0

        return [childA, childB]

    def __arithmeticCross(self, selection, generational=True, a=0.5):
        child = []

        for j in range(0, self.length):
            gene = (self.partial_new_generation[selection[0]][j] + self.partial_new_generation[selection[1]][j]) / 2

            if gene > self.upper_bound:
                gene = self.upper_bound
            elif gene < self.lower_bound:
                gene = 0.0

            child.append(gene)

        childs = [child]

        if generational:
            parentA = self.__fitnessCalculation(self.partial_new_generation[selection[0]], a)
            parentB = self.__fitnessCalculation(self.partial_new_generation[selection[1]], a)
            if parentA >= parentB:
                childs.append(self.partial_new_generation[selection[
                    0]])  # Puesto que tenemos un deficit de hijos, nos quedamos con uno de los padres en cada cruce
            else:
                childs.append(self.partial_new_generation[selection[
                    1]])  # Puesto que tenemos un deficit de hijos, nos quedamos con uno de los padres en cada cruce

        return childs

    def __mutationOperator(self, gene, desviacion=0.3):
        mut = np.random.normal(0, desviacion, 1)  # Gaussiana
        gene = gene + mut

        if gene > self.upper_bound:
            gene = self.upper_bound
        elif gene < self.lower_bound:
            gene = 0.0

        return gene

    def __bestWorseIndividual(self, alpha, best=True, how_many=1, new_generation=False):
        if not new_generation:
            ordened = [(self.__fitnessCalculation(i, alpha), i) for i in
                       self.actual_generation]  # Calcula el fitness de cada individuo, y lo guarda en pares ordenados siendo la primera componente la valoración y la segunda individuo
            ordened = [i[1] for i in sorted(ordened, reverse=True, key=itemgetter(
                0))]  # Ordena los pares ordenados (de más fitness a menos fitness) y nos quedamos con los individuos
            indexes = []
            for i in range(0, how_many):
                if best:
                    indexes.append(self.actual_generation.index(ordened[i]))
                else:
                    indexes.append(self.actual_generation.index(ordened[len(ordened) - (i + 1)]))
        else:
            ordened = [(self.__fitnessCalculation(i, alpha), i) for i in
                       self.partial_new_generation]  # Calcula el fitness de cada individuo, y lo guarda en pares ordenados siendo la primera componente la valoración y la segunda individuo
            ordened = [i[1] for i in sorted(ordened, reverse=True, key=itemgetter(
                0))]  # Ordena los pares ordenados (de más fitness a menos fitness) y nos quedamos con los individuos
            indexes = []
            for i in range(0, how_many):
                if best:
                    indexes.append(self.partial_new_generation.index(ordened[i]))
                else:
                    indexes.append(self.partial_new_generation.index(ordened[len(ordened) - (i + 1)]))

        return indexes

    def __localSearch(self, prob=1.0, variance=0.3, alpha=0.5, improved=False, bound=2):
        cromosome_bound = round(prob * self.size)
        bound = bound * self.length

        if prob == 1.0:
            for z in range(0, cromosome_bound):
                original_evaluation = self.__fitnessCalculation(self.partial_new_generation[z], alpha)
                iterations = 0

                while iterations < bound:
                    indexes = list(range(0,
                                         self.length))  # Pesos que vamos a modificar para intentar alcanzar una solución más prometedora
                    np.random.shuffle(indexes)

                    for j in indexes:
                        old_gene = self.partial_new_generation[z][j]  # Guardamos el gen por si no mejora
                        new_gene = self.__mutationOperator(self.partial_new_generation[z][j], variance)
                        self.partial_new_generation[z][j] = new_gene
                        new_evaluation = self.__fitnessCalculation(self.partial_new_generation[z], alpha)
                        iterations += 1

                        if new_evaluation > original_evaluation:  # Se ha encontrado un mejor vecino por lo que salimos y volvemos a comenzar la BL sobre este nuevo vecino
                            original_evaluation = self.__fitnessCalculation(self.partial_new_generation[z], alpha)
                            break
                        else:
                            self.partial_new_generation[z][j] = old_gene

        elif improved:
            individuals = self.__bestWorseIndividual(alpha, best=True, how_many=cromosome_bound, new_generation=True)
            np.random.shuffle(individuals)

            for z in range(0, cromosome_bound):
                original_evaluation = self.__fitnessCalculation(self.partial_new_generation[individuals[z]], alpha)
                iterations = 0

                while iterations < bound:
                    indexes = list(range(0,
                                         self.length))  # Pesos que vamos a modificar para intentar alcanzar una solución más prometedora
                    np.random.shuffle(indexes)

                    for j in indexes:
                        old_gene = self.partial_new_generation[z][j]  # Guardamos el gen por si no mejora
                        new_gene = self.__mutationOperator(self.partial_new_generation[individuals[z]][j], variance)
                        self.partial_new_generation[z][j] = new_gene
                        new_evaluation = self.__fitnessCalculation(self.partial_new_generation[individuals[z]], alpha)
                        iterations += 1

                        if new_evaluation > original_evaluation:  # Se ha encontrado un mejor vecino por lo que salimos y volvemos a comenzar la BL sobre este nuevo vecino
                            original_evaluation = self.__fitnessCalculation(self.partial_new_generation[z], alpha)
                            break
                        else:
                            self.partial_new_generation[individuals[z]][j] = old_gene

        else:
            individuals = list(range(0, cromosome_bound))
            np.random.shuffle(individuals)

            for z in range(0, cromosome_bound):
                original_evaluation = self.__fitnessCalculation(self.partial_new_generation[individuals[z]], alpha)
                iterations = 0

                while iterations < bound:
                    indexes = list(range(0,
                                         self.length))  # Pesos que vamos a modificar para intentar alcanzar una solución más prometedora
                    np.random.shuffle(indexes)

                    for j in indexes:
                        old_gene = self.partial_new_generation[z][j]  # Guardamos el gen por si no mejora
                        new_gene = self.__mutationOperator(self.partial_new_generation[individuals[z]][j], variance)
                        self.partial_new_generation[z][j] = new_gene
                        new_evaluation = self.__fitnessCalculation(self.partial_new_generation[individuals[z]], alpha)
                        iterations += 1

                        if new_evaluation > original_evaluation:  # Se ha encontrado un mejor vecino por lo que salimos y volvemos a comenzar la BL sobre este nuevo vecino
                            original_evaluation = self.__fitnessCalculation(self.partial_new_generation[z], alpha)
                            break
                        else:
                            self.partial_new_generation[individuals[z]][j] = old_gene

    def generationalModel(self, alpha=0.0, beta=0.5, mut_variance=0.3, local_search=0, prob_LS=0.0, improved_LS=False):
        self.partial_new_generation = self.actual_generation[:]

        if self.iteLS == 0.0 and prob_LS != 0.0:
            self.iteLS = local_search

        while self.actual_iteration < self.iterations:
            actualBest = self.__bestWorseIndividual(beta, best=True)  # ELITISM
            overwritten_best = False

            if alpha == 0.0:
                for i in range(0, self.pressure):
                    selection = [self.__binaryTournament(beta), self.__binaryTournament(beta)]
                    cross = self.__arithmeticCross(selection=selection, a=beta)
                    self.partial_new_generation[selection[0]] = cross[0]
                    self.partial_new_generation[selection[1]] = cross[1]
                    if selection[0] == actualBest or selection[1] == actualBest:
                        overwritten_best = True
            else:
                for i in range(0, self.pressure):
                    selection = [self.__binaryTournament(beta), self.__binaryTournament(beta)]
                    childs = self.__crossOperatorBLX(selection, alpha)
                    self.partial_new_generation[selection[0]] = childs[0]
                    self.partial_new_generation[selection[1]] = childs[1]
                    if selection[0] == actualBest or selection[1] == actualBest:
                        overwritten_best = True

            # Mutation

            for i in range(0, self.number_mutation):
                chromosome = np.random.randint(0, self.size)
                gene = np.random.randint(0, self.length)
                self.partial_new_generation[chromosome][gene] = self.__mutationOperator(
                    self.partial_new_generation[chromosome][gene], mut_variance)

            if overwritten_best:
                actualWorse = self.__bestWorseIndividual(beta, best=False,
                                                         new_generation=True)  # Elitismo, la peor solución de la nueva generación se sustituye por la mejor de la anterior
                self.partial_new_generation[actualWorse[0]] = self.actual_generation[actualBest[0]]

            if self.enterLS:
                self.enterLS = False
                self.__localSearch(prob_LS, mut_variance, alpha=beta,
                                   improved=improved_LS)  # Coindice la varianza con la varianza de mutación, devuelve las iteraciones que ha tomado la BL

            self.actual_generation = self.partial_new_generation[:]
        #            print(self.__fitnessCalculation(self.actual_generation[self.__bestWorseIndividual(beta)[0]], alpha = beta))     # Comprobar ELITISMO -> CORRECTO

        return self.actual_generation[self.__bestWorseIndividual(beta)[0]]

    def stationaryModel(self, alpha=0.0, beta=0.5, mut_variance=0.3):
        self.partial_new_generation = self.actual_generation[:]

        while self.actual_iteration < self.iterations:
            selection = [self.__binaryTournament(beta), self.__binaryTournament(beta)]

            if alpha == 0.0:
                cross = self.__arithmeticCross(selection=selection, a=beta)
            else:
                cross = self.__crossOperatorBLX(selection, alpha)

            worse = self.__bestWorseIndividual(beta, best=False, how_many=2)
            evaluation_w = []
            evaluation_w.append(self.__fitnessCalculation(self.partial_new_generation[worse[0]], beta))
            evaluation_w.append(self.__fitnessCalculation(self.partial_new_generation[worse[1]], beta))
            evaluation_n = []
            evaluation_n.append(self.__fitnessCalculation(cross[0], beta))
            evaluation_n.append(self.__fitnessCalculation(cross[1], beta))

            if evaluation_w[0] < evaluation_n[0]:
                self.partial_new_generation[worse[0]] = cross[0]
            elif evaluation_w[1] < evaluation_n[1]:
                self.partial_new_generation[worse[1]] = cross[1]

            # Mutation

            for i in range(0, self.number_mutation):
                chromosome = np.random.randint(0, len(selection))
                gene = np.random.randint(0, self.length)
                self.partial_new_generation[selection[chromosome]][gene] = self.__mutationOperator(
                    self.partial_new_generation[selection[chromosome]][gene], mut_variance)

            self.actual_generation = self.partial_new_generation[:]
        #            print(self.__fitnessCalculation(self.__bestWorseIndividual(beta, best = True), beta))

        return self.actual_generation[self.__bestWorseIndividual(beta, best=True)[0]]
