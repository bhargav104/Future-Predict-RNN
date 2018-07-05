import torch
import math
import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from generator import generate_copying_sequence
from tensorboardX import SummaryWriter
import torchvision as T

#writer = SummaryWriter()

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
hid_size = 100
out_size = 10
lr = 0.001
train_size = 50000
test_size = 10000
update_fq = 20
ktrunc = 50

class Net(nn.Module):
	
	def __init__(self, inp_size, hid_size, out_size):
		super().__init__()
		self.lstm = nn.LSTM(inp_size, hid_size)
		self.fc1 = nn.Linear(hid_size, out_size)

	def forward(self, x, state):
		x, new_state = self.lstm(x, state)
		x = self.fc1(x)
		return x, new_state	

def test_model(model, criterion):
	
	accuracy = 0
	loss = 0
	with torch.no_grad():
		for i, data in enumerate(testloader, 1):
			test_x, test_y = data
			test_x = test_x.view(-1, 784, 1)
			test_x, test_y = test_x.to(device), test_y.to(device)
			test_x.transpose_(0, 1)
			h = torch.zeros(1, batch_size, hid_size).to(device)
			c = torch.zeros(1, batch_size, hid_size).to(device)

			for j in range(T):
				output, (h, c) = model(test_x[j].unsqueeze(0), (h, c))

			loss += criterion(output[0], test_y).item()
			preds = torch.argmax(output[0], dim=1)
			correct = preds == test_y
			accuracy += correct.sum().item()

	accuracy /= 100.0
	loss /= 100.0

	print('test loss ' + str(loss) + ' accuracy ' + str(accuracy))
	return loss, accuracy



def train_model(model, epochs, criterion, optimizer):

	best_acc = 0.0
	ctr = 0	
	global lr
	for epoch in range(epochs):
		print('epoch ' + str(epoch + 1))
		epoch_loss = 0

		if epoch % update_fq == update_fq - 1:
			lr = lr / 2.0
			optimizer.lr = lr

		for z, data in enumerate(trainloader, 0):
			inp_x, inp_y = data
			inp_x = inp_x.view(-1, 28*28, 1)
			inp_x, inp_y = inp_x.to(device), inp_y.to(device)
			inp_x.transpose_(0, 1)
			h = torch.zeros(1, batch_size, hid_size).to(device)
			c = torch.zeros(1, batch_size, hid_size).to(device)
				
			sq_len = T
			loss = 0

			for i in range(sq_len):
				
				#if i % ktrunc == ktrunc - 1 and i != sq_len - 1:
				#	h = h.detach()
				#	c = c.detach()
				output, (h, c) = model(inp_x[i].unsqueeze(0), (h, c))
			
			loss += criterion(output[0], inp_y)

			model.zero_grad()
			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), 5.0)
			optimizer.step()

			loss_val = loss.item()
			print(z, loss_val)
			#writer.add_scalar('/MNIST', loss_val, ctr)
			ctr += 1

		t_loss, accuracy = test_model(model, criterion)
		best_acc = max(best_acc, accuracy)
		print('best accuracy ' + str(best_acc))
		#writer.add_scalar('/accMNIST', accuracy, epoch)

device = torch.device('cuda')
net = Net(inp_size, hid_size, out_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)

train_model(net, n_epochs, criterion, optimizer)
#writer.close()