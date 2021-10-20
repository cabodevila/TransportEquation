import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import re

"""
This script takes the output files from the program and creates an animation of the time evolution of each parameter
"""

class Animate():

    def __init__(self, time_steps = False):

        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 12))

        self.lattice = np.loadtxt('plattice.txt')
        self.fun_arx = sorted(os.listdir('data/function'), key=lambda x: int(re.search('iteration_(.*).txt', x).group(1)))
        self.num_arx = sorted(os.listdir('data/number'), key=lambda x: int(re.search('iteration_(.*).txt', x).group(1)))
        self.integrals = np.loadtxt('integrals.txt')
        self.stats = np.loadtxt('stats.txt')
        self.index = [re.search('iteration_(.*).txt', f).group(1) for f in self.fun_arx]

        self.line1, = self.axes[0, 0].plot(self.lattice, np.loadtxt('data/function/' + self.fun_arx[0]))
        self.line2, = self.axes[0, 1].plot(self.lattice, np.loadtxt('data/number/' + self.num_arx[0]))

        self.line30, = self.axes[1, 0].plot([0], self.integrals[:0+1,0], label=r'$I_a$')
        self.line31, = self.axes[1, 0].plot([0], self.integrals[:0+1,1], label=r'$I_b$')
        self.line32, = self.axes[1, 0].plot([0], self.integrals[:0+1,2], label=r'$T^*$')

        self.line40, = self.axes[1, 1].plot([0], self.stats[:0+1,0], label='total number')
        self.line41, = self.axes[1, 1].plot([0], self.stats[:0+1,1], label='total energy')
        self.line42, = self.axes[1, 1].plot([0], self.stats[:0+1,2], label='total entropy')

        self.axes[0, 0].set_title('Function distribution')
        self.axes[0, 1].set_title('Number distribution')
        self.axes[1, 0].set_title('Integrals')
        self.axes[1, 1].set_title('Stats')

        if time_steps:
            self.time_steps = time_steps
        else:
            self.time_steps = len(self.fun_arx)

        return

    def next(self, i):

        print('Iteration %i' %i)

        next_fun = np.loadtxt('data/function/' + self.fun_arx[i])
        next_num = np.loadtxt('data/number/' + self.num_arx[i])
        index = self.index[i]

        self.fig.suptitle('Iteration %i' %int(index))

        self.line1.set_ydata(next_fun)
        self.axes[0, 0].set_ylim([-0.1*max(next_fun), 1.1*max(next_fun)])
        self.axes[0, 0].grid()
        #self.axes[0, 0].legend(loc='upper right')

        self.line2.set_ydata(next_num)
        self.axes[0, 1].set_ylim([-0.1*max(next_num), 1.1*max(next_num)])
        self.axes[0, 1].grid()
        #self.axes[0, 1].legend(loc='upper right')

        self.line30.set_ydata(self.integrals[:i+1,0])
        self.line30.set_xdata(range(len(self.integrals[:i+1,0])))
        self.line31.set_ydata(self.integrals[:i+1,1])
        self.line31.set_xdata(range(len(self.integrals[:i+1,1])))
        self.line32.set_ydata(self.integrals[:i+1,2])
        self.line32.set_xdata(range(len(self.integrals[:i+1,2])))
        self.axes[1,0].set_xlim([-1, i+1])
        self.axes[1, 0].legend(loc='upper left')
        self.axes[1, 0].grid()

        self.line40.set_ydata(self.stats[:i+1,0])
        self.line40.set_xdata(range(len(self.stats[:i+1,0])))
        self.line41.set_ydata(self.stats[:i+1,1])
        self.line41.set_xdata(range(len(self.stats[:i+1,1])))
        self.line42.set_ydata(self.stats[:i+1,2])
        self.line42.set_xdata(range(len(self.stats[:i+1,2])))
        self.axes[1,1].set_xlim([-1, i+1])
        self.axes[1, 1].legend(loc='upper left')
        self.axes[1, 1].grid()

        return self.line1, self.line2, self.line30, self.line31, self.line32, self.line40, self.line41, self.line42,

    def animate(self):

        animation = FuncAnimation(self.fig, self.next, interval=20, blit=True, save_count=self.time_steps)
        animation.save('anim.mp4')