"""
This code computes the time evolution of the number of particles in each momentum volume
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import Latticev2 as Lattice
import scipy.interpolate as scpi

class Evolution():

    def __init__(self, deltat, nt_steps, pmin, pmax, size, Qs = 0.1):

        self.deltat  = deltat
        self.nt      = nt_steps
        self.np      = size

        self.lattice = Lattice.Lattice(pmin, pmax, size)
        self.lattice.save_lattice()

        # Space between each element of the lattice
        self.deltap = np.array([self.lattice.get_lattice()[i+1] - self.lattice.get_lattice()[i]
                                for i in range(len(self.lattice.get_lattice()) - 1)])

        # Space between each mean value of the lattice
        self.deltap_ = np.array([self.lattice.get_p_lattice()[i+1] - self.lattice.get_p_lattice()[i]
                                 for i in range(len(self.lattice.get_p_lattice()) - 1)])

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
                #sigma = 0.03
                #func.append(0.1 * np.exp(-(p - self.qs)**2/(2*sigma**2)))

        func = np.array(func)
        numb = np.array([self.lattice.number(i, func[1:]) for i in range(len(self.lattice.get_p_lattice())-1)])
        numb = np.append(numb, 0)

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
            - numpy array with the current
        """

        # Useful quantities
        """
        Ia = np.trapz(self.lattice.get_lattice()**2 * self.function * (1 + self.function), self.lattice.get_lattice()) / (2 * np.pi ** 2)
        Ib = 2 * np.trapz(self.lattice.get_lattice() * self.function, self.lattice.get_lattice()) / (2 * np.pi ** 2)
        T_star = Ia / Ib
        """

        Ia = np.trapz(self.lattice.get_lattice()**2 * self.function * (1 + self.function), x= self.lattice.get_lattice()) / (2 * np.pi ** 2)
        Ib = 2 * np.trapz(self.lattice.get_lattice() * self.function, x= self.lattice.get_lattice()) / (2 * np.pi ** 2)
        T_star = Ia / Ib

        # Debye screening mass
        #mD2 = 2 * self.alphas * self.Nc * Ib / np.pi

        # Jet quenching parameter
        #p_t = 1
        #self.qhat = 4 * self.alphas**2 * self.Nc**2 * Ia * np.log(p_t / mD2) / np.pi
        self.qhat = Ia

        #mom_deriv = np.array([(self.function[i+1] - self.function[i]) / self.deltap_[i] for i in range(len(self.deltap_))])
        #mom_deriv = (self.function[1:] - self.function[:-1]) / self.deltap

        #Using a second order derivative
        mom_deriv = (self.function[2:] - self.function[:-2]) / (self.deltap[:-1] + self.deltap[1:])
        #mom_deriv = np.array([(self.function[i+1] - self.function[i-1]) / (self.deltap_[i-1] + self.deltap_[i]) for i in range(1, len(self.deltap_))]) # Second order derivative
        #mom_deriv = np.insert(mom_deriv, 0, (self.function[1] - self.function[0]) / self.deltap_[0])

        Jp = self.qhat * (mom_deriv + self.function[1:-1] * (1 + self.function[1:-1]) / T_star) / 4
        Jp = np.append(Jp, 0) # Suppose at p = inf, Jp = 0
        Jp = np.insert(Jp, 0, 0)
        self.Jp = Jp

        """Regular interpolation"""
        #Jp_ = np.interp(self.lattice.get_p_lattice(),
        #                self.lattice.get_lattice(),
        #                Jp)

        """BSpline interpolation"""
        #tck = interpolate.splrep(self.lattice.get_p_lattice(), Jp)
        #spline  = interpolate.BSpline(*tck)

        #Jp_ = spline(self.lattice.get_lattice()[1:-1])
        #Jp_ = np.insert(Jp_, 0, 0) # No current at origin
        #Jp_ = np.append(Jp_, 0)

        deriv = (self.lattice.get_lattice()[1:]**2 * Jp[1:] - self.lattice.get_lattice()[:-1]**2 * Jp[:-1]) / (6 * np.pi**2)
        self.Jp = Jp
        self.deriv = deriv

        return deriv, Ia, Ib, T_star, Jp

    def next_step(self):

        """
        Update the next time step distribution values
        :return:
            - numpy array with the new distribution function values
            - numpy array with the new number density values
            - list containing the values of Ia, Ib, T_star the current Jp and the derivative deriv for the current time step.
        """

        # Compute the new number distribution and add the boundary condition n(infinity) = f(infinity) = 0
        deriv, Ia, Ib, T_star, Jp = self.derivative()
        number_new = self.number + self.deltat * deriv

        # Recover the new values for the function distribution
        function_new = number_new[1:] * (6 * np.pi**2) / (self.lattice.get_p_lattice()[1:]**3 - self.lattice.get_p_lattice()[:-1]**3)
        #function_new = np.append(function_new, function_new[-1])
        extrapolate1 = scpi.InterpolatedUnivariateSpline(self.lattice.get_lattice()[1:50], function_new[:49])
        self.function = np.insert(function_new, 0, extrapolate1(self.lattice.get_lattice()[0]))
        extrapolate2 = scpi.InterpolatedUnivariateSpline(self.lattice.get_lattice()[-50:-1], function_new[-49:])
        self.function = np.append(self.function, extrapolate2(self.lattice.get_lattice()[-1]))
        #function_new = np.insert(function_new, 0, extrapolate(self.lattice.get_lattice()[0]))

        #function_new = np.array(
        #    [number_new[i] / (self.lattice.get_lattice()[i+1] ** 3 - self.lattice.get_lattice()[i] ** 3)
        #     for i in range(len(number_new))]) * (6 * np.pi**2)
        #function_new = np.append(function_new, function_new[-1])

        self.number = number_new

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
                print(i)
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


    def thermal(self, p, mu, T):
        return 1 / (np.exp((p - mu) / T) - 1)

    def plot_evo(self, iter):

        if iter % 10000 == 0:
            plt.clf()

            #time_deriv = (self.Jp[1:] - self.Jp[:-1]) / self.deltap[1:-1]
            #plt.plot(time_deriv)
            #plt.xscale('log')
            #print(np.trapz(self.lattice.get_lattice()**2 * self.Jp))
            print(sum(self.deriv))
            plt.plot(self.lattice.get_lattice(), self.Jp*1e3)
            #plt.plot(self.lattice.get_p_lattice(), self.number*1e6, '.-')

            #plt.plot(self.function - self.function_old, 'r.', label='Distribution')
            plt.plot(self.lattice.get_lattice(), self.lattice.get_lattice() * self.function, 'r.', label='Distribution')

            #plt.plot(self.lattice.get_p_lattice(), self.lattice.get_p_lattice() * self.function_, 'b.', label='Distribution')

            mu_theo = -0.00691035
            T_theo  = 0.0267218
            plt.plot(self.lattice.get_lattice(), self.lattice.get_lattice() * self.thermal(self.lattice.get_lattice(), mu_theo, T_theo), 'k--', label='Theoretical distribution')

            plt.xscale('log')
            #plt.xlim(-0.005, 0.01)
            #plt.ylim([-1e-10, 1e-10])
            #plt.plot(self.number, 'b', label='Number')

            stats = np.loadtxt('stats.txt')
            energy = stats[:,1]
            plt.grid()
            plt.legend(loc='upper right', title='Iteration: %i\nMinimum = %e\nEnergy difference = %e' %(iter, min(self.function), energy[-1] - energy[0]))
            plt.pause(0.1)
            #plt.show()

        return
