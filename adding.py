import torch
import math
import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from generator import generate_adding_sequence
from tensorboardX import SummaryWriter

#writer = SummaryWriter()

torch.manual_seed(100)
np.random.seed(100)
tensor = torch.FloatTensor

n_epochs = 300
T = 10
batch_size = 100
inp_size = 1
hid_size = 128
out_size = 1
lr = 0.0001
train_size = 100000
test_size = 5000
update_fq = 100000
ktrunc = 5

def create_dataset(size, T):
	d_x = []
	d_y = []
	for i in range(size):
		sq_x, sq_y = generate_adding_sequence(T)
		sq_x, sq_y = sq_x, sq_y
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
	inp_y = test_y
	h = torch.zeros(1, test_size, hid_size).to(device)
	c = torch.zeros(1, test_size, hid_size).to(device)
		
	with torch.no_grad():
		for i in range(T + T):
			output, (h, c) = model(inp_x[i].unsqueeze(0), (h, c))

		loss += criterion(output[0], inp_y).item()


	print('test loss ' + str(loss))
	return loss



def train_model(model, epochs, criterion, optimizer):

	train_x, train_y = create_dataset(train_size, T)
	test_x, test_y = create_dataset(test_size, T)
	train_x, train_y = train_x.to(device), train_y.to(device)
	test_x, test_y = test_x.to(device), test_y.to(device)
	best_loss = 1e6
	ctr = 0	
	global lr
	for epoch in range(epochs):
		print('epoch ' + str(epoch + 1))
		epoch_loss = 0

		#if epoch % update_fq == update_fq - 1:
		#	lr = lr / 2.0
		#	optimizer.lr = lr

		for z in range(train_size // batch_size):
			ind = np.random.choice(train_size, batch_size)
			inp_x, inp_y = train_x[ind], train_y[ind]
			inp_x.transpose_(0, 1)
			h = torch.zeros(1, batch_size, hid_size).to(device)
			c = torch.zeros(1, batch_size, hid_size).to(device)
				
			sq_len = T * 2
			loss = 0

			for i in range(sq_len):
				
				#if i % ktrunc == ktrunc - 1 and i != sq_len - 1:
				#	h = h.detach()
				#	c = c.detach()

				output, (h, c) = model(inp_x[i].unsqueeze(0), (h, c))

			#print(output, inp_y)
			loss += criterion(output[0], inp_y)
			
			model.zero_grad()
			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()

			loss_val = loss.item()
			print(z, loss_val)
			#writer.add_scalar('/adding200', loss_val, ctr)
			ctr += 1

		t_loss = test_model(model, test_x, test_y, criterion)
		best_loss = min(best_loss, t_loss)
		print('best loss ' + str(best_loss))
		#writer.add_scalar('/adding200test', t_loss, epoch)

device = torch.device('cuda')
net = Net(inp_size, hid_size, out_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=lr)

train_model(net, n_epochs, criterion, optimizer)
#writer.close()
