import torch
import math
import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from generator import generate_sequence

torch.manual_seed(100)
np.random.seed(100)
tensor = torch.FloatTensor

n_epochs = 500
sq_len = 100
batch_size = 1
inp_size = 2
hid_size = 64

class Net(nn.Module):
	
	def __init__(self, inp_size, hid_size, out_size):
		super().__init__()
		self.lstm = nn.LSTM(inp_size, hid_size)
		self.fc1 = nn.Linear(hid_size, out_size)

	def forward(self, x, state):
		x, new_state = self.lstm(x, state)
		x = self.fc1(x)
		return x, new_state	


def train_model(model, epochs, criterion, optimizer):

	for epoch in range(epochs):
		epoch_loss = 0
		
		for _ in range(100):
			inp_x, inp_y = generate_sequence(sq_len)
			inp_x.transpose_(0, 1)
			h = torch.zeros(1, batch_size, hid_size)
			c = torch.zeros(1, batch_size, hid_size)
			state = (h, c)
			
			for step in inp_x:
				output, state = model(step.unsqueeze(0), state)

			output = output.squeeze(0)
			loss = criterion(output, inp_y)
			model.zero_grad()
			loss.backward()
			optimizer.step()

			loss_val = loss.item()
			print(loss_val)
			print(output, inp_y)


net = Net(inp_size, hid_size, inp_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

train_model(net, n_epochs, criterion, optimizer)