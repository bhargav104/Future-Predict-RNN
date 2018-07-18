import torch
import math
import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lstm_cell import LSTM
from generator import generate_onebit_sequence
from tensorboardX import SummaryWriter

#writer = SummaryWriter()

torch.manual_seed(100)
np.random.seed(100)
tensor = torch.FloatTensor

n_epochs = 1000
T = 100
batch_size = 100
inp_size = 1
hid_size = 1
out_size = 1
lr = 0.01
train_size = 100000
test_size = 5000
update_fq = 10000000
ktrunc = 5

def create_dataset(size, T):
	d_x = []
	d_y = []
	for i in range(size):
		sq_x, sq_y = generate_onebit_sequence(T)
		sq_x, sq_y = sq_x[0], sq_y[0]
		d_x.append(sq_x)
		d_y.append(sq_y)

	d_x = torch.stack(d_x)
	d_y = torch.stack(d_y)
	return d_x, d_y


class Net(nn.Module):
	
	def __init__(self, inp_size, hid_size, out_size):
		super().__init__()
		self.lstm = LSTM(inp_size, hid_size)
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
	h = torch.zeros(test_size, hid_size).to(device)
	c = torch.zeros(test_size, hid_size).to(device)
		
	for i in range(T):
		output, (h, c) = model(inp_x[i], (h, c))
	loss += criterion(output, inp_y.t()).item()
	'''
	preds = torch.argmax(output[0], dim=1)
	actual = inp_y.t()
	correct = preds == actual
	accuracy += correct.sum().item()

	accuracy /= (50.0)
	'''
	print('test loss ' + str(loss) + ' accuracy ' + str(accuracy))
	return loss, accuracy

def train_model(model, epochs, criterion, optimizer):

	train_x, train_y = create_dataset(train_size, T)
	test_x, test_y = create_dataset(test_size, T)
	train_x, train_y = train_x.to(device), train_y.to(device)
	test_x, test_y = test_x.to(device), test_y.to(device)
	best_acc = 0.0
	ctr = 0	
	global lr
	for epoch in range(epochs):
		print('epoch ' + str(epoch + 1))
		epoch_loss = 0

		if epoch % update_fq == update_fq - 1:
			lr = lr / 2.0
			optimizer.lr = lr

		for z in range(train_size // batch_size):
			ind = np.random.choice(train_size, batch_size)
			inp_x, inp_y = train_x[ind], train_y[ind]
			inp_x.transpose_(0, 1)
			inp_y.transpose_(0, 1)
			h = torch.zeros(batch_size, hid_size).to(device)
			c = torch.zeros(batch_size, hid_size).to(device)
				
			sq_len = T
			loss = 0

			p_trunc = 0.2
			val = np.random.random(size=1)[0]
			for i in range(sq_len):
				
				#if i % ktrunc == ktrunc - 1 and i != sq_len - 1:
				#	h = h.detach()
				#	c = c.detach()

				output, (h, c) = model(inp_x[i], (h, c))

			loss += criterion(output, inp_y.t())

			model.zero_grad()
			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()

			loss_val = loss.item()	
			print(z, loss_val)
			#writer.add_scalar('/400change', loss_val, ctr)
			ctr += 1

		t_loss, accuracy = test_model(model, test_x, test_y, criterion)
		if accuracy > best_acc:
			best_acc = accuracy
			#torch.save(model.state_dict(), 'copy100full.pt')
		print('best accuracy ' + str(best_acc))
		#writer.add_scalar('/acc400change', accuracy, epoch)

device = torch.device('cpu')
net = Net(inp_size, hid_size, out_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=lr)

train_model(net, n_epochs, criterion, optimizer)
#writer.close()