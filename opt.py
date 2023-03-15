# -*- coding: utf-8 -*-



import numpy as np
from collections import namedtuple


Result = namedtuple('Result', ('nfev', 'cost', 'gradnorm', 'x')) # число, массив, число, вектор


def gauss_newton(y, f, j, x0, k=0.5, tol=1e-4):
    """Метод Гаусса-Ньютона
    
    y -  массив измерений
    f(*x) - модельная функция, возващает вектор значений с размером y.shape
    j(*x) - якобиан, возвращает матрицу с размером (y.size - строки, x0.size - столбцы)
    x0 - вектор с начальным приближением
    k - параметр метода
    tol - относительная ошибка x (tolerance)
    
    Возвращает объект класса Result
    Документация вызывается help(f), ?f или f.__doc__
    """
    nfev = 0
    costArray = []
    x = np.asarray(x0, dtype=np.float) 
    while True:
        r = y - f(*x)
        nfev += 1
        cost = 0.5 * np.dot(r, r)
        costArray.append(cost)
        
        jac = j(*x)
        g = np.dot(jac.T, r) # J^T(y-t); np.dot или @ - матричное умножение
        delta_x = np.linalg.solve(np.dot(jac.T, jac), g) # решаем СЛУ
        x += k * delta_x
        norm_deltaX = np.linalg.norm(delta_x)
        norm_X = np.linalg.norm(x)
        if norm_deltaX <= tol * norm_X:
            break
    
    return Result(nfev, costArray, gradnorm=np.linalg.norm(g), x=x)


def lm(y, f, j, x0, lmbd0=1e-2, nu=2, tol=1e-4):
    
    nfev = 0
    costArray = []
    x = np.asarray(x0, dtype=np.float)
    lmbd = lmbd0
    while True:
        r = y - f(*x)
        nfev += 1
        cost = 0.5 * np.dot(r,r) # Ф(k-1) - cost с предыдущего шага 
        jac = j(*x)
        g = np.dot(jac.T, r)
        JtJ = np.dot(jac.T, jac)
        I = np.eye(JtJ.shape[0], JtJ.shape[1])
        
        delta_X = np.linalg.solve(np.add(JtJ, (lmbd/nu)*I), g)
        xNextNu = x + delta_X
        rNextNu = y - f(*xNextNu)
        nfev += 1
        costNextNu = 0.5*np.dot(rNextNu,rNextNu) # Ф(л(k-1)/nu) - cost с уменьшенной лямбдой
        if costNextNu <= cost: # Ф(л(k-1)/nu) <= Ф(k-1)
            x = xNextNu
            lmbd = lmbd/nu
            costArray.append(costNextNu)
        else:
            delta_X = np.linalg.solve(np.add(JtJ, lmbd*I), g) 
            xNext = x + delta_X
            rNext = y - f(*xNext)
            nfev += 1
            costNext = 0.5*np.dot(rNext,rNext) # Ф(л(k-1)) - cost с неизмененной/предыдущей лямбдой
            if costNextNu > cost and costNextNu <= costNext: # Ф(л(k-1)/nu) > Ф(k-1) and Ф(л(k-1)/nu) <= Ф(л(k-1))
                x = xNext
                costArray.append(costNext)
            else: # оставшийся случай: costNextNu > cost and costNextNu > costNext
                lmbd = lmbd*nu
                delta_X = np.linalg.solve(np.add(JtJ, lmbd*I), g)
                xNextNultiply = x + delta_X
                rNextNuMultiply = y - f(*xNextNultiply)
                nfev += 1
                costNextNuMultiply = 0.5*np.dot(rNextNuMultiply,rNextNuMultiply)
                
                while (costNextNuMultiply > cost):
                    lmbd = lmbd*nu
                    delta_X = np.linalg.solve(np.add(JtJ, lmbd*I), g)
                    xNextNultiply = x + delta_X
                    rNextNuMultiply = y - f(*xNextNultiply)
                    nfev += 1
                    costNextNuMultiply = 0.5*np.dot(rNextNuMultiply,rNextNuMultiply)
                
                costArray.append(costNextNuMultiply)
                
        if np.linalg.norm(delta_X) <= tol*(np.linalg.norm(x)):
            break     
        
    return Result(nfev, costArray, gradnorm=np.linalg.norm(g), x=x)