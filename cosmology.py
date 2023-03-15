# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
import opt  # importing module with optimization algorithms
import json



def func (z, H_0, Omega):
    integrand = lambda x: 1/np.sqrt((1 - Omega) * ((1 + x)**3) + Omega)
    integral = np.fromiter((integrate.quad(integrand, 0, x)[0] for x in z ), dtype=np.float)
    value = 5*np.log10(3e+11/H_0 * (1+z) * integral) - 5
    return value


def j (z, H_0, Omega):
    jac = np.empty((z.size, 2), dtype=np.float)
    jac[:, 0] = -5/(H_0 * np.log(10)) # производная по H_0
    integrand = lambda x: 1/np.sqrt((1 - Omega) * ((1 + x)**3) + Omega)
    integral = np.fromiter((integrate.quad(integrand, 0, x)[0] for x in z ), dtype=np.float)
    diffIntegrand = lambda x: (1 - (x+1)**3)/np.power(Omega + ((x+1)**3) * (1-Omega), 1.5)
    diffIntegral = np.fromiter((integrate.quad(diffIntegrand, 0, x)[0] for x in z ), dtype=np.float)
    jac[:, 1] = -5/(2*np.log(10)) * diffIntegral/integral # производная по Omega
            
    return jac


def main():
    with open('jla_mub.txt','r',encoding='utf-8-sig') as file:
        values = [line.split(' ') for line in file if line]
        redshift = np.asarray([x[0] for x in values], dtype=np.float) # size 31
        distance = np.asarray([x[1] for x in values], dtype=np.float)
       
    H0 = 50
    Om = 0.5
    approximation = (H0, Om)
    
    gnModel = opt.gauss_newton(
        distance, 
        lambda *x: func(redshift, *x), 
        lambda *x: j(redshift, *x), 
        approximation, 
        k=1, 
        tol=1e-4)
    
    lmModel = opt.lm(
            distance,
            lambda *x: func(redshift, *x),
            lambda *x: j(redshift, *x), 
            approximation, 
            lmbd0=1e-2, 
            nu=5,
            tol=1e-4)
    
    print(gnModel)
    print(lmModel)

    plt.figure(num=None, figsize=(10, 6), dpi=120)
    plt.plot(redshift, distance, 'x', label='data')
    plt.plot(redshift, func(redshift, *gnModel.x), label='GN model')
    plt.plot(redshift, func(redshift, *lmModel.x), label='LM model')
    plt.legend()
    plt.xlabel('Красное смещение')
    plt.ylabel('Модуль расстояния')
    plt.savefig('mu-z.png')
    
    plt.figure(num=None, figsize=(10, 6), dpi=120)
    gnCostRange = np.arange(1, len(gnModel.cost)+1, 1)
    lmCostRange = np.arange(1, len(lmModel.cost)+1, 1)
    plt.plot(gnCostRange, gnModel.cost, 'x', label='GN model cost')
    plt.plot(lmCostRange, lmModel.cost, '.', label='LM model cost')
    plt.legend()
    plt.xlabel('Итерация')
    plt.ylabel('Cost')
    plt.savefig('cost.png')
    
    data = {"Gauss-Newton": {"H0": gnModel.x[0], "Omega": gnModel.x[1], "nfev": gnModel.nfev},
            "Levenberg-Marquardt": {"H0": lmModel.x[0], "Omega": lmModel.x[1], "nfev": lmModel.nfev}}
    with open('parameters.json', 'w', encoding='utf8') as file:
        json.dump(data,file)


if __name__ == '__main__':
    main()
    

























































