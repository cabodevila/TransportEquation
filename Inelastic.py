import numpy as np
import scipy.interpolate as scpi
import matplotlib.pyplot as plt

from dask import delayed, compute
import multiprocessing as mp

class inelastic_kernel():

    def __init__(self, function, Ia, Ib, lattice, p_lattice, xlims=[1e-3, 1-1e-3]):

        self.alphas = 0.1
        self.Nc = 3

        self.Ia = Ia
        self.Ib = Ib
        self.T  = Ia / Ib

        # Debye screening mass
        self.mD2 = 2 * self.alphas * self.Nc * Ib / np.pi
        self.pt2 = 1

        self.lattice = lattice
        self.p_lattice = p_lattice
        self.function = function * lattice

        self.lattice_aux = np.append(lattice, 2 * lattice[-1] - lattice[-2])
        self.function_aux = np.append(self.function, 0)

        self.extr = scpi.InterpolatedUnivariateSpline(self.lattice_aux, self.function_aux, ext=3)
        self.pf = self.extr(p_lattice)

        self.x = np.linspace(*xlims, 5000)

        return
    """
    def extr(self, p):
        return np.interp(p, self.lattice_aux, self.function_aux)"""

    def split_rate(self):

        """
        Computes the rate of a hard gluon with momentum p \sim Q_s to split almost
        collinearly into two gluons with momenta px and p(1-x)
        """

        qhat = 8 * np.pi * self.alphas**2 * self.Nc**2 * self.Ia #* np.log(self.pt2 / self.mD2)

        h0 = (self.alphas * self.Nc / np.pi) * np.sqrt(qhat)
        h = (1-self.x+self.x**2)**(5/2) / (self.x-self.x**2)**(3/2)

        return h0 * h # The 1/sqrt(p) is added later

    def new_grid(self, p):

        self.factors = [1/self.x, (1-self.x)/self.x, self.x, 1-self.x]
        self.new_f = []

        for i, factor in enumerate(self.factors):
            new_grid = factor * p
            self.new_f.append(self.extr(new_grid) / factor)

        return

    def integrand(self, p, i):

        self.new_grid(p)

        Ca = (self.new_f[0] * (p + self.pf[i]) * (p + self.new_f[1]) -
             self.pf[i] * self.new_f[1] * (p + self.new_f[0])) / self.x**(5/2)

        Cb = (self.pf[i] * (p + self.new_f[2]) * (p + self.new_f[3]) -
             self.new_f[2] * self.new_f[3] * (p + self.pf[i]))

        integrand = np.trapz(self.split_rate() * (Ca - Cb * np.heaviside(self.x-0.5, 0)), self.x)

        return integrand

    def derivative(self):

        lattice_aux = np.copy(self.p_lattice)
        lattice_aux[0] = self.lattice[0]

        # Execute the integration in x in parallel
        pool = mp.Pool(6)
        lista = [[p, i] for i, p in enumerate(lattice_aux)]
        integrand = np.array(pool.starmap(self.integrand, lista))
        integrand = integrand * lattice_aux**(-3/2)

        # Integrate in momentum in order to compute the derivative
        derivative = np.array(
            [np.trapz(integrand[i:i+2], x=self.p_lattice[i:i+2])
            for i in range(1, len(self.lattice))]
            )

        derivative = np.insert(derivative,
                               0,
                               integrand[1] * self.p_lattice[1])

        derivative = derivative / (2 * np.pi**2)

        return derivative
