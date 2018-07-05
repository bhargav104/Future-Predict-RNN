import torch
import math
import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from generator import generate_copying_sequence
from tensorboardX import SummaryWriter

#writer = SummaryWriter()

torch.manual_seed(400)
np.random.seed(400)
tensor = torch.FloatTensor

n_epochs = 500
T = 200
batch_size = 100
inp_size = 1
hid_size = 128
out_size = 9
lr = 0.001
train_size = 100000
test_size = 5000
update_fq = 50
ktrunc = 5

def create_dataset(size, T):
	d_x = []
	d_y = []
	for i in range(size):
		sq_x, sq_y = generate_copying_sequence(T)
		sq_x, sq_y = sq_x[0], sq_y[0]
		d_x.append(sq_x)
		d_y.append(sq_y)

	d_x = torch.stack(d_x)
	d_y = torch.stack(d_y)
	return d_x, d_y


class Net(nn.Module):
	
	def __init__(self, inp_size, hid_size, out_size):
		super().__init__()
		self.lstm = nn.LSTM(inp_size, hid_size)
		self.fc1 = nn.Linear(hid_size, out_size)

	def forward(self, x, state):
		x, new_state = self.lstm(x, state)
		x = self.fc1(x)
		return x, new_state	

def test_model(model, test_x, test_y, criterion):
	loss = 0
	accuracy = 0
	inp_x = torch.t(test_x)
	inp_y = torch.t(test_y)
	h = torch.zeros(1, test_size, hid_size).to(device)
	c = torch.zeros(1, test_size, hid_size).to(device)

	with torch.no_grad():	
		for i in range(T + 20):
			output, (h, c) = model(inp_x[i].unsqueeze(0), (h, c))
			x1 = c[0][1].norm().item()
			x2 = c[0][1].mean().item()
			x3 = c[0][1].max().item()
			x4 = c[0][1].min().item()
			print(x1,x2,x3,x4)
			'''
			writer.add_scalar('/transfer-norm', x1, i)
			writer.add_scalar('/transfer-mean', x2, i)
			writer.add_scalar('/transfer-max', x3, i)
			writer.add_scalar('/transfer-min', x4, i)
			'''
			#print(c[0][0])
			loss += criterion(output[0], inp_y[i].squeeze(1)).item()
			if i >= T + 10:
				preds = torch.argmax(output[0], dim=1)
				actual = inp_y[i].squeeze(1)
				correct = preds == actual
				accuracy += correct.sum().item()

	loss /= (T + 20.0)
	accuracy /= (500.0)

	print('test loss ' + str(loss) + ' accuracy ' + str(accuracy))
	return loss, accuracy

device = torch.device('cuda')
net = Net(inp_size, hid_size, out_size).to(device)
net.load_state_dict(torch.load('copy100trunc.pt'))
criterion = nn.CrossEntropyLoss()
test_x, test_y = create_dataset(test_size, T)	
test_x, test_y = test_x.to(device), test_y.to(device)

test_model(net, test_x, test_y, criterion)	
#writer.close()