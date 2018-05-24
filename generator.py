import numpy as np
from zfilter import ZFilter
import torch

zf = ZFilter((2,))
tensor = torch.FloatTensor

def generate_sequence(sq_len, train=True):
	# make velocities/radius random
	#vx, vy = np.random.uniform(-1, 1, 1)[0], np.random.uniform(-1, 1, 1)[0]
	vx, vy = 0.5, 0.5
	# initial position randomly select on circle of radius size
	start_x = np.random.uniform(-sq_len, 0, 1)[0]
	start_y = np.random.uniform(-sq_len, 0, 1)[0]

	x = []
	i = 0
	cur_x, cur_y = start_x, start_y
	frac = np.random.uniform(0.25, 0.75, 1)[0] * sq_len
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