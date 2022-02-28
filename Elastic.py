import numpy as np
import scipy.interpolate as scpi

class elastic_kernel():

    def __init__(self, function, Ia, Ib, lattice, p_lattice):

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
        self.function = function

    def derivative(self):

        # Jet quenching parameter
        qhat = 8 * np.pi * self.alphas**2 * self.Nc**2 * self.Ia #* np.log(self.pt2 / self.mD2)
        #qhat = self.Ia

        #Using a second order derivative

        pf = self.lattice * self.function
        mom_deriv = (pf[1:] - pf[:-1]) / (self.lattice[1:] - self.lattice[:-1])

        extr_ = scpi.InterpolatedUnivariateSpline(self.lattice, pf)
        pf_ = extr_(self.p_lattice[1:-1])
        #pf_ = np.interp(self.p_lattice[1:-1], self.lattice, pf)

        deriv = self.p_lattice[1:-1]*(mom_deriv-pf_/self.p_lattice[1:-1])
        ff = pf_ * (self.p_lattice[1:-1] + pf_) / self.T

        Jp_ = qhat * (deriv + ff) / (4 * self.p_lattice[1:-1]**2)
        Jp_ = np.insert(Jp_, 0, 0)
        Jp_ = np.append(Jp_, 0)

        deriv = (self.p_lattice[1:]**2 * Jp_[1:] - self.p_lattice[:-1]**2 * Jp_[:-1]) / (2 * np.pi**2)

        return deriv, Jp_
