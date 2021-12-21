import numpy as np
from scipy.optimize import least_squares

def polylog(s, z):

    suma = 0

    for i in range(1, 10000):
        suma += (z**i) / (i**s)

    return suma

def equations(alpha):

    f0 = 0.1

    eq = f0*32/3 - polylog(3, np.exp(alpha))**4 / polylog(4, np.exp(alpha))**3

    return eq

def e2(alpha):

    f0 = 0.1

    eq = 1 - (6 * polylog(3, np.exp(alpha)))**(1/3) / (24 * polylog(4, np.exp(alpha)))**(1/4)

    return eq

def equations2(alpha, sigma, Qs):

    f0 = 0.1
    sigma = sigma / Qs

    A = Qs * (1/3 + np.sqrt(np.pi/2)*sigma + 2*sigma**2 + np.sqrt(np.pi/2)*sigma**3)**(1/3)
    B = Qs * (1/4 + np.sqrt(np.pi/2)*sigma + 3*sigma**2 + 3*np.sqrt(np.pi/2)*sigma**3 + 2*sigma**4)**(1/4)

    eq = (27*f0/2) * (A/B)**12 - polylog(3, np.exp(alpha))**4 / polylog(4, np.exp(alpha))**3

    return eq

def solve(x0, Qs=0.1):

    alpha = least_squares(equations, x0).x
    T = Qs * 0.25 * polylog(3, np.exp(alpha)) / polylog(4, np.exp(alpha))
    mu = alpha * T

    return alpha, T, mu

def solve2(x0, sigma=1/np.sqrt(20), Qs=0.1):

    f0 = 0.1
    alpha = least_squares(equations2, x0, args=(sigma, Qs)).x
    sigma = sigma / Qs

    A = Qs * (1/3 + np.sqrt(np.pi/2)*sigma + 2*sigma**2 + np.sqrt(np.pi/2)*sigma**3)**(1/3)
    B = Qs * (1/4 + np.sqrt(np.pi/2)*sigma + 3*sigma**2 + 3*np.sqrt(np.pi/2)*sigma**3 + 2*sigma**4)**(1/4)

    T = (B**4 / A**3) * (polylog(3, np.exp(alpha)) / polylog(4, np.exp(alpha))) * (1/3)
    mu = alpha * T

    return alpha, T, mu
