"""
This code computes the time evolution of the number of particles in each momentum volume
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import Lattice

class Evolution():

    def __init__(self, deltat, nt_steps, pmin, pmax, size, Qs = 0.1):

        self.deltat  = deltat
        self.nt      = nt_steps
        self.np      = size

        self.lattice = Lattice.Lattice(pmin, pmax, size)
        self.lattice.save_lattice()

        # Space between each element of the lattice
        self.deltap = [self.lattice.get_lattice()[i+1] - self.lattice.get_lattice()[i]
                       for i in range(len(self.lattice.get_lattice()) - 1)]

        # Space between each mean value of the lattice
        self.deltap_ = [self.lattice.get_p_lattice()[i+1] - self.lattice.get_p_lattice()[i]
                        for i in range(len(self.lattice.get_p_lattice()) - 1)]

        self.qs = Qs
        self.alphas = 0.1 # Estimation for high momentum particles
        self.Nc = 3

        self.function, self.number = self.initial_condition()

        return

    def initial_condition(self):

        """
        Set the initial values of the distribution with a simple model of a step-function with saturation momentum Qs
        :return: two numpy arrays:
            - Function distribution on each volume
            - Number distribution on each volume
        """

        func = []
        for p in self.lattice.get_p_lattice():
            if p <= self.qs:
                # Take 0.1 as the amplitude to avoid Bose-Einstein condensation
                func.append(0.1)
            else:
                func.append(0)
                #func.append(0.1 * np.exp(-1000*(p - self.qs)**2))

        func = np.array(func)
        numb = np.array([self.lattice.number(i, func) for i in range(len(self.lattice.get_lattice())-1)])

        return func, numb

    def derivative(self):

        """
        Computes the value of the time derivative for the number distribution except for the last one for the current
        time step
        :return:
            - numpy array with the derivatives
            - float with the Ia integral result
            - float with the Ib integral result
            - float with the T_start result
        """

        # Useful quantities

        Ia = sum(self.deltap * self.lattice.get_p_lattice()**2 * self.function * (1 + self.function)) / (2 * np.pi ** 2)
        
        
        
        Ib = 2 * sum(self.deltap * self.lattice.get_p_lattice() * self.function) / (2 * np.pi ** 2)
        
        
        T_star = Ia / Ib

        # Debye screening mass
        mD2 = 2 * self.alphas * self.Nc * Ib / np.pi

        # Jet quenching parameter
        #p_t = 1
        #self.qhat = 4 * self.alphas**2 * self.Nc**2 * Ia * np.log(p_t / mD2) / np.pi
        self.qhat = Ia

        mom_deriv = np.array([(self.function[i+1] - self.function[i]) / self.deltap_[i] for i in range(len(self.function) - 1)])

        deriv = self.qhat * self.lattice.get_p_lattice()[:-1]**2 * (
                mom_deriv + self.function[:-1] * (1 + self.function[:-1]) / T_star) / 4

        return deriv, Ia, Ib, T_star

    def next_step(self):

        """
        Update the next time step distribution values
        :return:
            - numpy array with the new distribution function values
            - numpy array with the new number density values
            - list containing the values of Ia, Ib and T_star for the current time step.
        """

        # Compute the new number distribution and add the boundary condition n(infinity) = f(infinity) = 0
        deriv, Ia, Ib, T_star = self.derivative()
        number_new = self.number[:-1] + self.deltat * deriv
        number_new = np.append(number_new, 0)

        # Recover the new values for the function distribution
        function_new = np.array(
            [number_new[i] * (6 * np.pi**2) / (self.lattice.get_lattice()[i+1] ** 3 - self.lattice.get_lattice()[i] ** 3)
             for i in range(len(number_new))])

        self.function = function_new
        self.number = number_new

        return function_new, number_new, [Ia, Ib, T_star]

    def evolve(self, save=False, plot=False):

        """
        Compute the time evolution of the number distribution
        :param save: int, number of iteration after which data is saved. If False, no data is saved
        :param plot: if True, distribution is plot after each time step
        :return:
        """

        for i in range(self.nt):

            fn, nn, adi = self.next_step()

            if save and i % save == 0:
                self.save(i, additional=adi)

            if plot:
                self.plot_evo(i)

        return


    def save(self, iter, additional=False):

        """
        Saves the data of the current step in different text files
        :param iter: current iteration of the evolution
        :param additional: additional parameters to save. Must be False or a [Ia, Ib, T_star] list
        :return:
        """

        os.makedirs('data/function', exist_ok=True)
        os.makedirs('data/number', exist_ok=True)

        np.savetxt('data/function/iteration_%i.txt' %iter, self.function)
        np.savetxt('data/number/iteration_%i.txt' %iter, self.number)

        if additional:

            Ia, Ib, T_star = additional
            number = sum(self.number)
            energy = sum([self.lattice.energy(i, self.function) for i in range(len(self.function))])
            entropy = sum([self.lattice.entropy(i, self.function) for i in range(len(self.function))])

            integrals = open('integrals.txt', 'a')
            stats = open('stats.txt', 'a')

            integrals.write('%.16f %.16f %.16f\n' %(Ia, Ib, T_star))
            stats.write('%.16f %.16f %.16f\n' %(number, energy, entropy))

            integrals.close()
            stats.close()

        return

    def conservation(self):

        """
        Test if the conservation of number and energy was achieved during the simulation
        :return:
        """

        stats = np.loadtxt('stats.txt')
        number = stats[:,0]
        energy = stats[:,1]

        output = open('conservation.txt', 'w')

        numb = max(number) - min(number)
        output.write('Particle number was conserved but %.16f particles, a %f %%\n' %(numb, 100*numb/max(number)))

        ener = max(energy) - min(energy)
        output.write('Energy density was conserved but %.16f particles, a %f %%\n' % (ener, 100*ener/max(energy)))

        output.close()

        return

    def plot_evo(self, iter):

        if iter % 10 == 0:
            plt.clf()
            plt.plot(self.function, 'r', label='Distribution')
            plt.plot(self.number, 'b', label='Number')
            plt.legend(loc='upper right', title='Iteration: %i' %iter)
            plt.pause(0.1)

        return

