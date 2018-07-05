import torch
import math
import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from char_iterator import PTBTextIterator
from tensorboardX import SummaryWriter
import argparse

parser = argparse.ArgumentParser(description='auglang parameters')
parser.add_argument('--full', action='store_true', default=False, help='Use full BPTT')
parser.add_argument('--trunc', type=int, default=5, help='size of H truncations')
parser.add_argument('--maxdep', type=int, default=5, help='max depth of nesting parenthesis')
parser.add_argument('--p-op', type=float, default=0.7, help='probability of opening bracket')

args = parser.parse_args()
writer = SummaryWriter()

torch.manual_seed(100)
np.random.seed(100)
tensor = torch.FloatTensor

dname = 'text8'
n_epochs = 300
T = 500
batch_size = 128
inp_size = 1
hid_size = 1000
out_size = 51
lr = 0.0001
train_size = 1409
test_size = 78
update_fq = 10
ktrunc = args.trunc
path = './' + dname
tr_rand = np.random.random(size=16000000)
vl_rand = np.random.random(size=1000000)
ts_rand = np.random.random(size=1000000)
tot = 0

class Net(nn.Module):
	
	def __init__(self, inp_size, hid_size, out_size):
		super().__init__()
		self.embed = nn.Embedding(out_size, 620)
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
		for _ in range(test_size):
			inp_x = torch.LongTensor(dset.next())
			if dset == 'valid':
				inp_x = augment(inp_x, vl_rand)
			else:
				inp_x = augment(inp_x, ts_rand)
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

	loss /= (sq_len * test_size * 1.0)
	print('valid loss', loss)
	return loss

def augment(x, arr):

	maxdep = args.maxdep
	p_op = args.p_op
	rc = 0
	for i in range(batch_size):
		ind = []
		for j in range(T):
			if x[i][j] == 0:
				ind.append(j)
		l = len(ind)
		#print(l)
		if l % 2 == 1:
			l -= 1
		depth = 0
		for j in range(l):
			val = arr[rc]
			rc += 1
			if ((val <= p_op and depth < maxdep) or depth == 0) and l - j  > depth:
				depth += 1

			else:
				x[i][ind[j]] = out_size - 1 
				depth -= 1

	return x


def train_model(model, epochs, criterion, optimizer):

	train = PTBTextIterator(which_data=dname, which_set='train', batch_size=128, shuffle_every_epoch=0, length=T, path=path)
	valid = PTBTextIterator(which_data=dname, which_set='valid', batch_size=128, shuffle_every_epoch=0, length=T, path=path)
	test = PTBTextIterator(which_data=dname, which_set='test', batch_size=128, shuffle_every_epoch=0, length=T, path=path)
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

		for z in range(train_size):
			inp_x = torch.LongTensor(train.next())
			inp_x = augment(inp_x, tr_rand)
			inp_x = inp_x.view(inp_x.size()[0], inp_x.size()[1], 1)
			inp_y = torch.cat((inp_x[:, 1:, :], torch.zeros([inp_x.size()[0], 1, 1], dtype=torch.long)), 1)
			inp_x, inp_y = inp_x.to(device), inp_y.to(device)
			inp_x.transpose_(0, 1)
			inp_y.transpose_(0, 1)
			h = torch.zeros(1, batch_size, hid_size).to(device)
			c = torch.zeros(1, batch_size, hid_size).to(device)
				
			sq_len = T - 1
			loss = 0
			# 3, 5, 10, 20, 30
			for i in range(sq_len):
				
				if i % ktrunc == ktrunc - 1 and i != sq_len - 1 and not args.full:
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
			writer.add_scalar('/augtexttrain', loss_val, ctr)
			ctr += 1

			if z % 500 == 499 or z == train_size - 1:
				t_loss = test_model(model, valid, criterion)
				valid.idx = 0
				valid.data2 = valid.chop()
				if t_loss < best_loss:
					best_loss = t_loss
					test_best = test_model(model, test, criterion)
					test.idx = 0
					test.data2 = test.chop()

				print('best loss ' + str(test_best))
				writer.add_scalar('/augtextvalid', t_loss, ctr1)
				ctr1 += 1


		train.idx = 0
		train.data2 = train.chop()


device = torch.device('cuda')
net = Net(inp_size, hid_size, out_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)

train_model(net, n_epochs, criterion, optimizer)
#writer.close()

'''
full, 30, 20, 10, 5, 3 1
augmented ptb - opt = full, trunc. maxdep = 30, 40, 50.
'''