import os
import re
import numpy as np
import matplotlib.pyplot as plt
import calculations

def names():

    os.makedirs('figures/function', exist_ok=True)
    os.makedirs('figures/number', exist_ok=True)

    lattice = np.loadtxt('lattice.txt')
    fun_arx = sorted(os.listdir('data/function'), key=lambda x: int(re.search('iteration_(.*).txt', x).group(1)))
    num_arx = sorted(os.listdir('data/number'), key=lambda x: int(re.search('iteration_(.*).txt', x).group(1)))
    deriv_elastic = sorted(os.listdir('data/deriv_elastic'), key=lambda x: int(re.search('iteration_(.*).txt', x).group(1)))
    deriv_inelastic = sorted(os.listdir('data/deriv_inelastic'), key=lambda x: int(re.search('iteration_(.*).txt', x).group(1)))
    integrals = np.loadtxt('integrals.txt')
    stats = np.loadtxt('stats.txt')
    chem_pot = np.loadtxt('chem_pot.txt')

    fun = [fun_arx[i] for i in range(len(fun_arx))]
    num = [num_arx[i] for i in range(len(num_arx))]

    return lattice, fun, num, integrals, stats, chem_pot, deriv_elastic, deriv_inelastic

def plot_fun(fun_names, lattice, deriv_elastic, deriv_inelastic):

    for i, name in enumerate(fun_names):
        plt.clf()
        plt.plot(lattice, lattice * np.loadtxt('data/function/' + name), label='distribution')
        plt.plot(lattice, 1e6 * lattice * np.loadtxt('data/deriv_elastic/' + deriv_elastic[i]), label=r'elastic $\times 10^6$')
        plt.plot(lattice, 1e6 * lattice * np.loadtxt('data/deriv_inelastic/' + deriv_inelastic[i]), label=r'inelastic $\times 10^6$')
        plt.xscale('log')
        plt.title(name)
        plt.legend()
        plt.grid()
        plt.savefig('figures/function/fun_' + name[:-4] + '.png')

    return

def plot_num(num_names, lattice):

    for name in num_names:
        plt.clf()
        plt.plot(lattice, np.loadtxt('data/number/' + name))
        plt.title(name)
        plt.grid()
        plt.savefig('figures/number/num_' + name[:-4] + '.png')

    return

def plot_integrals(data, time_step, steps, data_save, stats):

    Ia = data[:,0]
    Ib = data[:,1]
    T = data[:,2]
    # Preditions elastic scenario
    alpha, T_theo, mu = calculations.solve(-0.2, Qs=0.1)

    # Predictions elastic-inelastic scenario
    ene = stats[-1,1]
    T_theo = (30 * ene / np.pi**2) ** (1/4)

    x_axis = np.linspace(0, time_step * steps, int(steps/data_save))

    plt.clf()
    plt.figure(figsize=(15,10))
    plt.hlines(T_theo, x_axis[0], x_axis[-1], colors='k', linestyles='dashed', label=r'$T_{theo} = %.3f$' %T_theo)
    plt.plot(x_axis, Ia, label=r'$I_a$')
    plt.plot(x_axis, Ib, label=r'$I_b$')
    plt.plot(x_axis, T, label=r'$T^*$')

    plt.grid()
    plt.legend(loc='lower left')
    plt.savefig('figures/integrals.png')

    return

def plot_stats(data, time_step, steps, data_save):

    num = data[:,0]
    ene = data[:,1]
    ent = data[:,2]

    x_axis = np.linspace(0, time_step * steps, int(steps/data_save))

    plt.clf()
    plt.figure(figsize=(15,10))
    plt.plot(x_axis, num, label='total number')
    plt.plot(x_axis, ene, label='total energy')
    plt.plot(x_axis, ent, label='total entropy')

    plt.grid()
    plt.legend()
    plt.savefig('figures/stats.png')

    return

def plot_energy(data, time_step, steps, data_save):

    num = data[:,0]
    ene = data[:,1]
    ent = data[:,2]

    x_axis = np.linspace(0, time_step * steps, int(steps/data_save))

    plt.clf()
    plt.figure(figsize=(15,10))
    plt.plot(x_axis, ene, label='total energy')

    plt.grid()
    plt.legend()
    plt.savefig('figures/energy.png')

    return

def plot_chemical_potential(chem_pot, time_step, steps, data_save):

    alpha, T, chem_theo = calculations.solve(-0.2, Qs=0.1)

    x_axis = np.linspace(0, time_step * steps, int(steps/data_save))

    plt.clf()
    plt.figure(figsize=(15,10))
    plt.plot(chem_pot, label=r'$\mu^*$')
    #plt.hlines(chem_theo, 0, len(chem_pot), colors='k', linestyles='dashed', label=r'$\mu_{theo} = %.3f$' %chem_theo)

    plt.grid()
    plt.legend()
    plt.savefig('figures/chem_pot.png')

    return

def plot_last_distribution(fun, lattice):

    f = np.loadtxt('data/function/' + fun[-1])

    ener0 = np.loadtxt('stats.txt')[0, 1]
    ener1 = np.loadtxt('stats.txt')[-1, 1]

    T_theo0 = (30 * ener0 / np.pi**2) ** (1/4)
    T_theo1 = (30 * ener1 / np.pi**2) ** (1/4)

    def thermal(p, T):
        return 1 / (np.exp(p / T) - 1)

    plt.clf()
    plt.figure(figsize=(15,10))

    plt.plot(lattice, lattice * f, 'b.', label='Distribution')
    plt.plot(lattice, lattice * thermal(lattice, T_theo0), 'k--', label='Theoretical distr t=0')
    plt.plot(lattice, lattice * thermal(lattice, T_theo1), 'g--', label='Theoretical distr t=inf')

    plt.xscale('log')
    plt.grid()
    plt.legend()
    plt.savefig('figures/last_distribution.png')

    return

def plot_fun2(fun_names, lattice, integrals):

    for i, name in enumerate(fun_names):
        plt.clf()
        plt.plot(lattice[5:50], np.loadtxt('data/function/' + name)[5:50])
        plt.plot(lattice[5:50], (integrals[i, 2]/lattice[5:50]), 'k--', label='Theo')
        plt.xscale('log')
        plt.legend()
        plt.title(name)
        plt.grid()
        plt.savefig('figures/function/fun_' + name[:-4] + '.png')

    return


lattice, fun, num, integrals, stats, chem_pot, deriv_elastic, deriv_inelastic = names()

time_step = 2e-3
steps = int(5e5)
data_save = 1000

plot_integrals(integrals, time_step, steps, data_save, stats)
plot_stats(stats, time_step, steps, data_save)
plot_chemical_potential(chem_pot, time_step, steps, data_save)
plot_last_distribution(fun, lattice)
plot_fun(fun, lattice, deriv_elastic, deriv_inelastic)
plot_num(num, lattice)
