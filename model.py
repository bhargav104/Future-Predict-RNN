import torch
import math
import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lstm_cell import LSTM
from generator import generate_sequence

torch.manual_seed(100)
np.random.seed(100)
tensor = torch.FloatTensor

n_epochs = 500
sq_len = 10
batch_size = 1
inp_size = 2
hid_size = 64

class Predictor(nn.Module):
	
	def __init__(self, inp_size, hid_size):
		super().__init__()
		self.fc1 = nn.Linear(inp_size + 1, hid_size)
		self.mean = nn.Linear(hid_size, inp_size)
		self.variance = nn.Linear(hid_size, inp_size)

	def forward(self, x):
		x = F.tanh(self.fc1(x))
		return self.mean(x), F.softplus(self.variance(x))

class Net(nn.Module):
	
	def __init__(self, inp_size, hid_size, out_size):
		super().__init__()
		self.lstm = LSTM(inp_size, hid_size)
		self.fc1 = nn.Linear(hid_size, out_size)

	def forward(self, x, state):
		x, new_state = self.lstm(x, state)
		x = self.fc1(x)
		return x, new_state	

def train_model(model, epochs, criterion, optimizer_1, optimizer_2):

	for epoch in range(epochs):
		epoch_loss = 0
		
		for _ in range(100):
			inp_x, inp_y = generate_sequence(sq_len)
			inp_x.transpose_(0, 1)
			h = torch.zeros(batch_size, hid_size)
			c = torch.zeros(batch_size, hid_size)
			past_states = []
			
			for i in range(sq_len):
				best_norm = 1e8
				best_id = -1
				past_states.append(h)
				
				for j in range(i):
					time_diff = tensor([[(i - j) / (1.0 * sq_len)]])
					inp = torch.cat((past_states[j].detach(), time_diff), dim=1)
					pred_mean, pred_var = predictor(inp)
					pred_mean, pred_var = pred_mean.detach(), pred_var.detach()
					norm = torch.norm(pred_var)

					if norm < best_norm:
						best_norm = norm
						best_id = j
						best_inp = inp

				if best_id != -1:
					pred_mean, pred_var = predictor(best_inp)
					pred_var = pred_var.detach()
					# look into different ways to combine past into present
					alpha = torch.min(0.5 + pred_var, torch.ones(1, hid_size))
					h = alpha * h + (1 - alpha) * pred_mean

				output, (h, c) = net(inp_x[i], (h,c))

			loss = criterion(output, inp_y)
			optimizer_1.zero_grad()
			loss.backward(retain_graph=True)
			optimizer_1.step()

			arr_x = []
			arr_y = []
			for i in range(sq_len):
				for j in range(i):
					time_diff = tensor([[(i - j) / (1.0 * sq_len)]])
					inp = torch.cat((past_states[j], time_diff), dim=1)
					arr_x.append(inp)
					arr_y.append(past_states[i])
			p_x = torch.stack(arr_x)
			p_y = torch.stack(arr_y)

			pred_mean, pred_var = predictor(p_x)
			p_loss = torch.log(2 * math.pi * pred_var) + torch.pow((pred_mean - p_y) / (pred_var + 1e-7), 2)
			p_loss = p_loss.mean()

			optimizer_2.zero_grad()
			p_loss.backward()
			optimizer_2.step()

			loss_val = loss.item()
			p_val = p_loss.item()
			print(loss_val, p_val)
			print(output, inp_y)


net = Net(inp_size, hid_size, inp_size)
predictor = Predictor(hid_size, 64)
criterion = nn.MSELoss()
optimizer_1 = optim.Adam(list(net.parameters()) + list(predictor.parameters()), lr=0.001)
optimizer_2 = optim.Adam(predictor.parameters(), lr=0.001)
train_model(net, n_epochs, criterion, optimizer_1, optimizer_2)