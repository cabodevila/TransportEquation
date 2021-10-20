import os
import shutil

def remove():

	try:

		shutil.rmtree('data')
		os.remove('conservation.txt')
		os.remove('integrals.txt')
		os.remove('lattice.txt')
		os.remove('plattice.txt')
		os.remove('stats.txt')

	except OSError:
		pass

	return