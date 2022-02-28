import Time_evolution as te
import numpy as np
import delete
import time
import os

start = time.time()

#delete.remove()

time_step = 2e-3
steps = int(5e5)
data_save = 1000

grid_elements = 100

ev = te.Evolution(time_step, steps, 0, 0.5, grid_elements)

ev.evolve(save=data_save, all=True, plot=False)

ev.conservation()

f = open("time.txt", "w")
f.write("Time of run : %f s" % (time.time() - start))
f.close()
print(time.time() - start, ' s')


time.sleep(60)
os.system('shutdown now')
