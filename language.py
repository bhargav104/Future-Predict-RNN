import torch
import math
import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from char_iterator import PTBTextIterator
from tensorboardX import SummaryWriter

writer = SummaryWriter()

torch.manual_seed(100)
np.random.seed(100)
tensor = torch.FloatTensor

n_epochs = 30
T = 180
batch_size = 128
inp_size = 1
hid_size = 2000
out_size = 27
lr = 0.0001
train_size = 100000
test_size = 5000
update_fq = 10
ktrunc = 1

class Net(nn.Module):
	
	def __init__(self, inp_size, hid_size, out_size):
		super().__init__()
		self.embed = nn.Embedding(50, 620)
		self.lstm = nn.LSTM(620, hid_size)
		self.fc1 = nn.Linear(hid_size, out_size)
		self.embed.weight.data.uniform_(-0.1, 0.1)

	def forward(self, x, state):
		x = self.embed(x)
		x.transpose_(0, 1)
		x, new_state = self.lstm(x, state)
		x = self.fc1(x)
		return x, new_state

def test_model(model, dset, criterion):

	loss = 0
	sq_len = T - 1

	with torch.no_grad():
		for _ in range(218):
			inp_x = torch.LongTensor(dset.next())
			inp_x = inp_x.view(inp_x.size()[0], inp_x.size()[1], 1)
			inp_y = torch.cat((inp_x[:, 1:, :], torch.zeros([inp_x.size()[0], 1, 1], dtype=torch.long)), 1)
			inp_x, inp_y = inp_x.to(device), inp_y.to(device)
			inp_x.transpose_(0, 1)
			inp_y.transpose_(0, 1)
			h = torch.zeros(1, batch_size, hid_size).to(device)
			c = torch.zeros(1, batch_size, hid_size).to(device)
				
			for i in range(sq_len):
				output, (h, c) = model(inp_x[i], (h, c))
				output = F.softmax(output[0], dim=1)
				vals = torch.gather(output, 1, inp_y[i])
				vals = -torch.log2(vals).mean().item()
				loss += vals

	loss /= (sq_len * 218)
	print('valid loss', loss)
	return loss


def train_model(model, epochs, criterion, optimizer):

	train = PTBTextIterator(which_data='text8', which_set='train', batch_size=128, shuffle_every_epoch=1, length=T, path='./text8')
	valid = PTBTextIterator(which_data='text8', which_set='valid', batch_size=128, shuffle_every_epoch=0, length=T, path='./text8')
	test = PTBTextIterator(which_data='text8', which_set='test', batch_size=128, shuffle_every_epoch=0, length=T, path='./text8')
	best_loss = 1e6
	test_best = 1e6
	ctr = 0	
	ctr1 = 0
	global lr
	for epoch in range(epochs):
		print('epoch ' + str(epoch + 1))
		epoch_loss = 0

		#if epoch % update_fq == update_fq - 1:
		#	lr /= 2.0
		#	optimizer.lr = lr

		train_batches = 3928
		for z in range(train_batches):
			inp_x = torch.LongTensor(train.next())
			inp_x = inp_x.view(inp_x.size()[0], inp_x.size()[1], 1)
			inp_y = torch.cat((inp_x[:, 1:, :], torch.zeros([inp_x.size()[0], 1, 1], dtype=torch.long)), 1)
			inp_x, inp_y = inp_x.to(device), inp_y.to(device)
			inp_x.transpose_(0, 1)
			inp_y.transpose_(0, 1)
			h = torch.zeros(1, batch_size, hid_size).to(device)
			c = torch.zeros(1, batch_size, hid_size).to(device)
				
			sq_len = T - 1
			loss = 0

			for i in range(sq_len):
				
				if i % ktrunc == ktrunc - 1 and i != sq_len - 1:
					h = h.detach()
				#	c = c.detach()

				output, (h, c) = model(inp_x[i], (h, c))
				loss += criterion(output[0], inp_y[i].squeeze(1))

			loss /= (1.0 * sq_len)

			model.zero_grad()
			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()

			loss_val = loss.item()
			print(z, loss_val)
			writer.add_scalar('/texttrain', loss_val, ctr)
			ctr += 1

			if z % 500 == 499 or z == train_batches - 1:
				t_loss = test_model(model, valid, criterion)
				valid.idx = 0
				valid.data2 = valid.chop()
				if t_loss < best_loss:
					best_loss = t_loss
					test_best = test_model(model, test, criterion)
					test.idx = 0
					test.data2 = test.chop()

				print('best loss ' + str(test_best))
				writer.add_scalar('/textvalid', t_loss, ctr1)
				ctr1 += 1


		train.idx = 0
		train.data2 = train.chop()


device = torch.device('cuda')
net = Net(inp_size, hid_size, out_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)

train_model(net, n_epochs, criterion, optimizer)
writer.close()

'''
full, 30, 20, 10, 5, 3 1
'''