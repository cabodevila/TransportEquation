import numpy as np

"""
Create a class which defines a lattice in which we will make our calculations.
It is based in the simple model of the only-dependent momentum collision
"""

class Lattice():

    def __init__(self, pmin, pmax, size, different_steps=1e-2):

        """
        :param pmin: minimum value of the momentum lattice
        :param pmax: maximum value of the momentum lattice
        :param size: number elements of the lattice if there are not a different step size
        :param different_steps: if float, this is the number before which the momentum grid step is reduced
        """

        self.pmin = pmin
        self.pmax = pmax

        step = pmax / size
        latt = [pmin]

        if different_steps:

            while latt[-1] <= pmax:
                if latt[-1] <= different_steps:
                    latt.append(latt[-1] + step*0.2)
                else:
                    latt.append(latt[-1] + step)

            self.p_lattice = np.array(latt)

        else:

            self.p_lattice = np.linspace(pmin, pmax, size)

        self.len_p_lattice = len(self.p_lattice)

        self.lattice = (self.p_lattice[:-1] + self.p_lattice[1:]) / 2


    def get_lattice(self):

        return self.lattice

    def get_p_lattice(self):

        return self.p_lattice

    def number(self, i, f):

        """
        Computes the number of particles in the volume of the phase space which momentum is in the interval
        (p_lattice[i], p_lattice[i+1])
        :param i: int, index of the lattice on which we want to compute the number
        :param f: function distribution of momentum
        :return: float
        """

        return (self.p_lattice[i+1]**3 - self.p_lattice[i]**3) * f[i] / (6 * np.pi**2)

    def energy(self, i, f):

        """
        Computes the energy density in the volume of the phase space which momentum is in the interval
        (p_lattice[i], p_lattice[i+1])
        :param i: int, index of the lattice on which we want to compute the number
        :param f: function distribution of momentum
        :return: float
        """

        return (self.p_lattice[i+1]**4 - self.p_lattice[i]**4) * f[i] / (8 * np.pi**2)

    def entropy(self, i, f):

        """
        Computes the entropy in the volume of the phase space which momentum is in the interval
        (p_lattice[i-1], p_lattice[i])
        Notice that in f=0, there is an indetermination in the term f*log(f), so we take the limit 0*log(0)=0
        :param i: int, index of the lattice on which we want to compute the number
        :param f: function distribution of momentum
        :return: float
        """

        if f[i] <= 0:
            return (self.p_lattice[i + 1] ** 3 - self.p_lattice[i] ** 3) * (1 + f[i]) * np.log(1 + f[i]) / (6 * np.pi ** 2)
        else:
            return (self.p_lattice[i + 1] ** 3 - self.p_lattice[i] ** 3) * (
                    (1 + f[i]) * np.log(1 + f[i]) - f[i] * np.log(f[i])) / (6 * np.pi ** 2)

    def save_lattice(self):

        """
        Save the lattice in a .txt file
        :return:
        """

        np.savetxt('lattice.txt', self.get_lattice())
        np.savetxt('plattice.txt', self.get_p_lattice())
