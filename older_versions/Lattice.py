import numpy as np

"""
Create a class which defines a lattice in which we will make our calculations.
It is based in the simple model of the only-dependent momentum collision
"""

class Lattice():

    def __init__(self, pmin, pmax, size):

        """
        :param pmin: minimum value of the momentum lattice
        :param pmax: maximum value of the momentum lattice
        :param size: number elements of the lattice
        """

        self.pmin = pmin
        self.pmax = pmax
        self.size = size

        self.lattice = np.linspace(pmin, pmax, size)
        self.len_lattice = len(self.lattice)
        self.p_lattice = np.array([(self.lattice[i] + self.lattice[i+1]) / 2 for i in range(self.len_lattice - 1)])


    def get_lattice(self):

        """
        Returns the array which contains the lattice
        :return: numpy.array
        """

        return self.lattice

    def get_p_lattice(self):

        """
        Returns the mean values for each interval of the lattice, which correspond with the momentum value of each
        volume
        :return: numpy.array
        """

        return self.p_lattice

    def number(self, i, f):

        """
        Computes the number of particles in the volume of the phase space which momentum is in the interval
        (p[i-1], p[i])
        :param i: int, index of the lattice on which we want to compute the number
        :param f: function distribution of momentum
        :return: float
        """

        return (self.lattice[i+1]**3 - self.lattice[i]**3) * f[i] / (6 * np.pi**2)

    def energy(self, i, f):

        """
        Computes the energy density in the volume of the phase space which momentum is in the interval
        (p[i-1], p[i])
        :param i: int, starting count on 1
        :param f: function distribution of momentum
        :return: float
        """

        return (self.lattice[i+1]**4 - self.lattice[i]**4) * f[i] / (8 * np.pi**2)

    def entropy(self, i, f):

        """
        Computes the entropy in the volume of the phase space which momentum is in the interval (p[i-1], p[i])
        Notice that in f=0, there is an indetermination in the term f*log(f), so we take the limit 0*log(0)=0
        :param i: int, starting count on 1
        :param f: function distribution of momentum
        :return: float
        """

        if f[i] == 0:
            return (self.lattice[i + 1] ** 3 - self.lattice[i] ** 3) * (1 + f[i]) * np.log(1 + f[i]) / (6 * np.pi ** 2)
        else:
            return (self.lattice[i + 1] ** 3 - self.lattice[i] ** 3) * (
                    (1 + f[i]) * np.log(1 + f[i]) - f[i] * np.log(f[i])) / (6 * np.pi ** 2)

    def save_lattice(self):

        """
        Save the lattice in a .txt file
        :return:
        """

        np.savetxt('lattice.txt', self.get_lattice())
        np.savetxt('plattice.txt', self.get_p_lattice())
