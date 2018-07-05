import numpy as np
from zfilter import ZFilter
import torch
import sys

zf = ZFilter((2,))
tensor = torch.FloatTensor

def generate_ball_sequence(sq_len, train=True):
	# make velocities/radius random
	#vx, vy = np.random.uniform(-1, 1, 1)[0], np.random.uniform(-1, 1, 1)[0]
	vx, vy = 0.5, 0.5
	# initial position randomly select on circle of radius size
	start_x = np.random.uniform(-sq_len, 0, 1)[0]
	start_y = np.random.uniform(-sq_len, 0, 1)[0]

	x = []
	i = 0
	cur_x, cur_y = start_x, start_y
	frac = np.random.uniform(0.25, 0.5, 1)[0] * sq_len
	frac = int(frac)
	
	# consider input to be given for first 3/4 th timesteps then 0s for remaining
	while i < frac:
		next_val = zf(np.array([cur_x, cur_y]), update=train)
		x.append(next_val)
		cur_x, cur_y = cur_x + vx, cur_y + vy
		i += 1

	# maybe make it random noise instead of 0s
	while i < sq_len:
		#next_val = zf(np.array([0.0, 0.0]), update=train)
		x.append([0.0, 0.0])
		cur_x, cur_y = cur_x + vx, cur_y + vy
		i += 1

	y = zf(np.array([cur_x, cur_y]), update=train)
	return tensor([x]), tensor([y])

def generate_copying_sequence(T):
	
	items = [1, 2, 3, 4, 5, 6, 7, 8, 0, 9]
	x = []
	y = []

	ind = np.random.randint(8, size=10)
	for i in range(10):
		x.append([items[ind[i]]])
	for i in range(T - 1):
		x.append([items[8]])
	x.append([items[9]])
	for i in range(10):
		x.append([items[8]])

	for i in range(T + 10):
		y.append([items[8]])
	for i in range(10):
		y.append([items[ind[i]]])

	x = np.array(x)
	y = np.array(y)

	return tensor([x]), torch.LongTensor([y])

def generate_adding_sequence(T):

	x = []
	sq = np.random.uniform(size=T)
	for i in range(T):
		x.append([sq[i]])
	fv = np.random.randint(0, T//2, 1)[0]
	sv = np.random.randint(T//2, T, 1)[0]
	for i in range(T):
		if i == fv or i == sv:
			x.append([1.0])
		else:
			x.append([0.0])

	y = tensor(np.array([sq[fv] + sq[sv]]))
	x = tensor(np.array(x))

	return x, y

def generate_onebit_sequence(T):
	arr = [-1.0, 1.0]
	x = []
	val = np.random.randint(2, size=1)[0]
	x.append([arr[val]])
	y = [arr[val]]
	for i in range(T - 1):
		val = np.random.normal(0, 0.2, 1)[0]
		x.append([val])

	return tensor([x]), torch.FloatTensor([y])