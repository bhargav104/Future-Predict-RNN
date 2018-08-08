import torch
import math
import sys
import numpy as np
import torch.nn as nn
from lstm_cell import LSTM
import torch.nn.functional as F
import torch.optim as optim
from generator import generate_copying_sequence
from tensorboardX import SummaryWriter
import torchvision as T
import argparse
import os
import glob
parser = argparse.ArgumentParser(description='sequential MNIST parameters')
parser.add_argument('--full', action='store_true', default=False, help='Use full BPTT')
parser.add_argument('--trunc', type=int, default=5, help='size of H truncations')
parser.add_argument('--p-full', type=float, default=0.0, help='probability of opening bracket')
parser.add_argument('--p-detach', type=float, default=1.0, help='probability of detaching each timestep')
parser.add_argument('--permute', action='store_true', default=False, help='pMNIST or normal MNIST')
parser.add_argument('--save-dir', type=str, default='default', help='save directory')
parser.add_argument('--cos', action='store_true', default=False, help='print cosine between consecutive updates')
parser.add_argument('--norms', action='store_true', default=False, help='Print gradient norms')
parser.add_argument('--ghg', action='store_true', default=False, help='print ghg values')
parser.add_argument('--lstm-size', type=int, default=100, help='width of LSTM')

args = parser.parse_args()

log_dir = 'exp/mnist/'+args.save_dir + '/'
if os.path.isdir(log_dir):
	print('deleting contents of experiment directory')
	for f in glob.glob(log_dir+'*'):
		print(f)
		os.remove(f)

writer = SummaryWriter(log_dir=log_dir)

torch.manual_seed(100)
np.random.seed(100)
tensor = torch.FloatTensor

trainset = T.datasets.MNIST(root='./MNIST', train=True, download=True, transform=T.transforms.ToTensor())
testset = T.datasets.MNIST(root='./MNIST', train=False, download=True, transform=T.transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, num_workers=2)

n_epochs = 200
T = 784
batch_size = 100
inp_size = 1
hid_size = args.lstm_size
out_size = 10
lr = 0.001
train_size = 50000
test_size = 10000
update_fq = 20
ktrunc = args.trunc

class Net(nn.Module):
	
	def __init__(self, inp_size, hid_size, out_size):
		super().__init__()
		self.lstm = LSTM(inp_size, hid_size)
		self.fc1 = nn.Linear(hid_size, out_size)

	def forward(self, x, state):
		x, new_state = self.lstm(x, state)
		x = self.fc1(x)
		return x, new_state	

def test_model(model, criterion, order):
	
	accuracy = 0
	loss = 0
	with torch.no_grad():
		for i, data in enumerate(testloader, 1):
			test_x, test_y = data
			test_x = test_x.view(-1, 784, 1)
			test_x, test_y = test_x.to(device), test_y.to(device)
			test_x.transpose_(0, 1)
			h = torch.zeros(batch_size, hid_size).to(device)
			c = torch.zeros(batch_size, hid_size).to(device)

			for j in order:
				output, (h, c) = model(test_x[j], (h, c))

			loss += criterion(output, test_y).item()
			preds = torch.argmax(output, dim=1)
			correct = preds == test_y
			accuracy += correct.sum().item()

	accuracy /= 100.0
	loss /= 100.0

	print('test loss ' + str(loss) + ' accuracy ' + str(accuracy))
	return loss, accuracy

def get_flat_grads(model):
	ret = []
	for param in model.parameters():
		ret.append(param.grad.data.view(-1))
	ret = torch.cat(ret, dim=0)
	return ret

def train_model(model, epochs, criterion, optimizer):

	best_acc = 0.0
	ctr = 0	
	global lr
	if args.permute:
		order = np.random.permutation(T)
	else:
		order = np.arange(T)
	
	cc = 0
	nc = 0
	old_grads = None
	
	for epoch in range(epochs):
		print('epoch ' + str(epoch + 1))
		epoch_loss = 0

		#if epoch % update_fq == update_fq - 1:
		#	lr = lr / 2.0
		#	optimizer.lr = lr
		for z, data in enumerate(trainloader, 0):
			inp_x, inp_y = data
			inp_x = inp_x.view(-1, 28*28, 1)
			inp_x, inp_y = inp_x.to(device), inp_y.to(device)
			inp_x.transpose_(0, 1)
			h = torch.zeros(batch_size, hid_size).to(device)
			c = torch.zeros(batch_size, hid_size).to(device)
			sq_len = T
			loss = 0

			for i in order:
				if args.p_detach != 1.0 and not args.full:
					val = np.random.random(size=1)[0]
					if val <= args.p_detach:
						h = h.detach()
				output, (h, c) = model(inp_x[i], (h, c))
			
			loss += criterion(output, inp_y)

			model.zero_grad()
			loss.backward()
			norms = nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()

			if args.norms:
				print('norms', norms)
				writer.add_scalar('/normsMNIST', norms, nc)
				nc += 1
			if args.cos:
				if old_grads is None:
					old_grads = get_flat_grads(model)
				else:
					new_grads = get_flat_grads(model)
					cos_val = ((old_grads * new_grads) / (torch.norm(old_grads) * torch.norm(new_grads))).sum().item()
					old_grads = new_grads.clone()
					print('cos val', cos_val)
					writer.add_scalar('/cosMNIST', cos_val, cc)
					cc += 1
			optimizer.step()

			loss_val = loss.item()
			print(z, loss_val)
			writer.add_scalar('/MNIST', loss_val, ctr)
			ctr += 1

		t_loss, accuracy = test_model(model, criterion, order)
		best_acc = max(best_acc, accuracy)
		print('best accuracy ' + str(best_acc))
		writer.add_scalar('/accMNIST', accuracy, epoch)

device = torch.device('cuda')
net = Net(inp_size, hid_size, out_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)

train_model(net, n_epochs, criterion, optimizer)
writer.close()

'''
MNISTstoch - full, p-detach = 0.9, 0.75, 0.5, 0.25, 0.1, no forget - full, trunc 20, p-detach = 0.05, 0.01, 0.4
pMNIST - same as above
'''