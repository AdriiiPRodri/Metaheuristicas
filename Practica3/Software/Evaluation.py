# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 12:46:28 2018

@author: adrianprodri
"""

import warnings

warnings.filterwarnings('ignore')  # Cuando acabe colocarlo para evitar warnings de deprecated


################ Funciones a evaluar ##########################

def tasa_red(ceros, caracteristicas):
    return (100 * (ceros / caracteristicas))


def evaluacion(a, prediccion, ceros, caracteristicas):
    return (a * prediccion * 100 + (1 - a) * tasa_red(ceros, caracteristicas))
