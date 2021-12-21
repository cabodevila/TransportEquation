"""
This code computes the time evolution of the number of particles in each momentum volume
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import Latticev4 as Lattice
import scipy.interpolate as scpi
import calculations as cal

class Evolution():

    def __init__(self, deltat, nt_steps, pmin, pmax, size, Qs = 0.1):

        self.deltat  = deltat
        self.nt      = nt_steps
        self.np      = size

        self.lattice = Lattice.Lattice(pmin, pmax, size)
        self.lattice.save_lattice()

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
        for p in self.lattice.get_lattice():
            if p <= self.qs:
                # Take 0.1 as the amplitude to avoid Bose-Einstein condensation
                func.append(0.1)
            else:
                #Uncomment this for a step function distribution
                func.append(0)

                #Uncomment this for a step function distribution with a smooth gaussian decay
                #self.sigma = 0.02
                #func.append(0.1 * np.exp(-(p - self.qs)**2/(2*sigma**2)))

        func = np.array(func)
        numb = np.array([self.lattice.number(i, func) for i in range(len(self.lattice.get_lattice()))])

        return func, numb

    def derivative(self):

        """
        Computes the value of the number distribution time derivative for the current time step
        :return:
            - numpy array with the derivatives
            - float with the Ia integral result
            - float with the Ib integral result
            - float with the T_start result
            - numpy array with the current
        """

        # Useful quantities
        Ia = np.trapz(self.lattice.get_lattice()**2 * self.function * (1 + self.function), x=self.lattice.get_lattice()) / (2 * np.pi ** 2)
        Ib = 2 * np.trapz(self.lattice.get_lattice() * self.function, x=self.lattice.get_lattice()) / (2 * np.pi ** 2)
        T_star = Ia / Ib

        self.T = T_star

        # Debye screening mass
        #mD2 = 2 * self.alphas * self.Nc * Ib / np.pi

        # Jet quenching parameter
        #p_t = 1
        #self.qhat = 4 * self.alphas**2 * self.Nc**2 * Ia * np.log(p_t / mD2) / np.pi
        self.qhat = Ia

        #Using a second order derivative
        mom_deriv = (self.function[1:] - self.function[:-1]) / (self.lattice.get_lattice()[1:] - self.lattice.get_lattice()[:-1])
        self.mom_deriv = mom_deriv

        # Extrapolate the value of the distribution function in the grid defined by get_p_lattice
        extr = scpi.InterpolatedUnivariateSpline(self.lattice.get_lattice(), self.function)
        self.function_ = extr(self.lattice.get_p_lattice())

        Jp_ = self.qhat * (mom_deriv + self.function_[1:-1] * (1 + self.function_[1:-1]) / T_star) / 4
        Jp_ = np.insert(Jp_, 0, 0)
        Jp_ = np.append(Jp_, 0)


        deriv = (self.lattice.get_p_lattice()[1:]**2 * Jp_[1:] - self.lattice.get_p_lattice()[:-1]**2 * Jp_[:-1]) / (6 * np.pi**2)
        self.Jp_ = Jp_
        self.deriv = deriv

        return deriv, Ia, Ib, T_star, Jp_

    def next_step(self):

        """
        Update the next time step distribution values
        :return:
            - numpy array with the new distribution function values
            - numpy array with the new number density values
            - list containing the values of Ia, Ib, T_star the current Jp and the derivative deriv for the current time step.
        """

        deriv, Ia, Ib, T_star, Jp = self.derivative()
        number_new = self.number + self.deltat * deriv

        # Recover the new values for the function distribution
        function_new = number_new * (6 * np.pi**2) / (self.lattice.get_p_lattice()[1:]**3 - self.lattice.get_p_lattice()[:-1]**3)

        self.number = number_new
        self.function = function_new

        return function_new, number_new, [Ia, Ib, T_star, Jp, deriv]

    def evolve(self, save=False, all=True, plot=False):

        """
        Compute the time evolution of the number distribution
        :param save: int, number of iteration after which data is saved. If False, no data is saved
        :param all: boolean, if True, it will save the values during the evolution
        :param plot: if True, distribution is plot after each time step
        :return:
        """

        plt.figure(figsize=(15, 10))
        for i in range(self.nt):

            fn, nn, adi = self.next_step()

            if save and i % save == 0:
                self.save(i, additional=adi, all=all)

            if plot:
                self.plot_evo(i)

        return

    def save(self, iter, additional=False, all=True):

        """
        Saves the data of the current step in different text files
        :param iter: current iteration of the evolution
        :param additional: additional parameters to save. Must be False or a [Ia, Ib, T_star, Jp, deriv] list
        :return:
        """

        if all and not additional:

            os.makedirs('data/function', exist_ok=True)
            os.makedirs('data/number', exist_ok=True)
            os.makedirs('data/Jp', exist_ok=True)
            os.makedirs('data/deriv', exist_ok=True)

            np.savetxt('data/function/iteration_%i.txt' %iter, self.function)
            np.savetxt('data/number/iteration_%i.txt' %iter, self.number)

        elif all and additional:

            os.makedirs('data/function', exist_ok=True)
            os.makedirs('data/number', exist_ok=True)
            os.makedirs('data/Jp', exist_ok=True)
            os.makedirs('data/deriv', exist_ok=True)

            np.savetxt('data/function/iteration_%i.txt' %iter, self.function)
            np.savetxt('data/number/iteration_%i.txt' %iter, self.number)

            Ia, Ib, T_star, Jp, deriv = additional
            number = sum(self.number)
            energy = sum([self.lattice.energy(i, self.function) for i in range(len(self.function)-1)])
            entropy = sum([self.lattice.entropy(i, self.function) for i in range(len(self.function)-1)])

            integrals = open('integrals.txt', 'a')
            stats = open('stats.txt', 'a')
            chem_pot = open('chem_pot.txt', 'a')

            integrals.write('%.16e %.16e %.16e\n' %(Ia, Ib, T_star))
            stats.write('%.16e %.16e %.16e\n' %(number, energy, entropy))
            chem_pot.write('%.16e\n' %(- T_star * np.log(1 + 1/self.function[0])))

            integrals.close()
            stats.close()
            chem_pot.close()

            np.savetxt('data/Jp/iteration_%i.txt' %iter, Jp)
            np.savetxt('data/deriv/iteration_%i.txt' %iter, deriv)

        elif not all and additional:

            Ia, Ib, T_star, Jp, deriv = additional
            number = sum(self.number)
            #energy = sum([self.lattice.energy(i, self.function) for i in range(len(self.function)-1)])
            energy = np.trapz(self.lattice.get_lattice()**3 * self.function, self.lattice.get_lattice()) / (2*np.pi**2)
            entropy = sum([self.lattice.entropy(i, self.function) for i in range(len(self.function)-1)])

            integrals = open('integrals.txt', 'a')
            stats = open('stats.txt', 'a')
            chem_pot = open('chem_pot.txt', 'a')

            integrals.write('%.16e %.16e %.16e\n' %(Ia, Ib, T_star))
            stats.write('%.16e %.16e %.16e\n' %(number, energy, entropy))
            chem_pot.write('%.16e\n' %(- T_star * np.log(1 + 1/self.function[0])))

            integrals.close()
            stats.close()
            chem_pot.close()

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
        output.write('Particle number was conserved but %.16e particles, a %.16f %%\n' %(numb, 100*numb/number[0]))

        ener = max(energy) - min(energy)
        output.write('Energy density was conserved but %.16e , a %.16f %%\n' % (ener, 100*ener/energy[0]))

        output.close()

        return

    def plot_evo(self, iter):

        """
        Plots the current function distribution and current and the theoretical form at equilibrium
        in order to follow the evolution during computation.
        """

        if iter % 10000 == 0:

            def thermal(p, mu, T):
                return 1 / (np.exp((p - mu) / T) - 1)

            plt.clf()

            plt.plot(self.lattice.get_p_lattice(), self.Jp_*1e3, 'r.-', label='Current')
            plt.plot(self.lattice.get_lattice(), self.lattice.get_lattice() * self.function, 'b.', label='Distribution')

            alpha, T_theo, mu_theo = cal.solve(-0.2, Qs=self.qs)
            plt.plot(self.lattice.get_lattice(), self.lattice.get_lattice() * thermal(self.lattice.get_lattice(), mu_theo, T_theo), 'k--', label='Theoretical distribution')

            plt.xscale('log')

            stats = np.loadtxt('stats.txt')
            energy = stats[:,1]
            plt.grid()
            plt.legend(loc='upper right', title='Iteration: %i' %iter)
            plt.pause(0.1)

        return
