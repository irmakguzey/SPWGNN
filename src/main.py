from TowerCreator import *


# This script reads the saved trajectory, trains the graph neural network, and 
if __name__ == '__main__':
	n = int(input('Please enter the number of rectangles you want: '))
	N = int(input('Please enter the number of iterations you want for this n: '))
	self_run_str = raw_input('Would you like to interract with the app or let the app run itself and have trajectory? [y/n]')
	print(self_run_str)

	if self_run_str == 'y':
		self_run = True
	else:
		self_run = False

	towerCreator = TowerCreator(self_run, n, N)
	towerCreator.run()