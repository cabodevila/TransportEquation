import Time_evolution as te
import Animate as an
import delete

delete.remove()

ev = te.Evolution(1e-2, 1500, 0, 0.2, 500)

ev.evolve(save=1)
ev.conservation()

an.Animate().animate()