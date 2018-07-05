import torch
import math
import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lstm_cell import LSTM
from generator import generate_copying_sequence
from tensorboardX import SummaryWriter

writer = SummaryWriter()

torch.manual_seed(100)
np.random.seed(100)
tensor = torch.FloatTensor

lr = 0.001
n_epochs = 300
T = 200
batch_size = 100
inp_size = 1
out_size = 9
hid_size = 128
ktrunc = 10
update_fq = 25
train_size = 100000
test_size = 5000
skip = 10

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

def test_model(model, test_x, test_y, criterion):
	loss = 0
	accuracy = 0
	inp_x = torch.t(test_x)
	inp_y = torch.t(test_y)
	h = torch.zeros(test_size, hid_size).to(device)
	c = torch.zeros(test_size, hid_size).to(device)
	past_states = []
	sq_len = T + 20
	with torch.no_grad():
		
		for i in range(T + 20):
			if i % skip == 0:
				past_states.append(h)

			norms = []
			for j in range(len(past_states)):
				time_diff = torch.ones(test_size, 1).to(device) * ((i - skip * j) / (1.0 * sq_len))
				inp = torch.cat((past_states[j], time_diff), dim=1)
				pred_mean, pred_var = predictor(inp)
				norm = torch.norm(pred_var, dim=1)
				norms.append(norm)

			if len(norms) != 0:
				norms = torch.stack(norms)
				top1 = torch.topk(norms, 1, dim=0)[1]
				best_inp = []
				best_td = []
				for j in range(test_size):
					best_inp.append(past_states[top1[0][j].item()][j])
					best_td.append([i - skip * top1[0][j].item()])
				best_td = tensor(best_td).to(device)
				best_inp = torch.stack(best_inp)
				best_inp_var = torch.cat((best_inp, best_td), dim=1)
				pred_mean, pred_var = predictor(best_inp_var)
				alpha = torch.min(0.5 + pred_var, torch.ones(test_size, hid_size).to(device))
				#h = alpha * h + (1 - alpha) * pred_mean
				h = alpha * h + (1 - alpha) * best_inp

			output, (h, c) = model(inp_x[i], (h, c))
			loss += criterion(output, inp_y[i].squeeze(1)).item()

			if i >= T + 10:
				preds = torch.argmax(output, dim=1)
				actual = inp_y[i].squeeze(1)
				correct = preds == actual
				accuracy += correct.sum().item()
	loss /= (T + 20.0)
	accuracy /= 500.0
	print('test loss ' + str(loss) + ' accuracy ' + str(accuracy))
	return loss, accuracy



def train_model(model, epochs, criterion, optimizer_1, optimizer_2):

	train_x, train_y = create_dataset(train_size, T)
	test_x, test_y = create_dataset(test_size, T)
	train_x, train_y = train_x.to(device), train_y.to(device)
	test_x, test_y = test_x.to(device), test_y.to(device)
	best_acc = 0.0
	global lr
	ctr = 0
	for epoch in range(epochs):
		epoch_loss = 0
		print('epoch ' + str(epoch + 1))
		
		for z in range(train_size // batch_size):

			ind = np.random.choice(train_size, batch_size)
			inp_x, inp_y = train_x[ind], train_y[ind]
			inp_x.transpose_(0, 1)
			inp_y.transpose_(0, 1)
			h = torch.zeros(batch_size, hid_size).to(device)
			c = torch.zeros(batch_size, hid_size).to(device)
			past_states = []
			
			sq_len = T + 20
			loss = 0

			for i in range(sq_len):
				
				best_norm = 1e8
				best_id = -1

				if i % ktrunc == ktrunc - 1 and i != sq_len - 1:
					h = h.detach()
					c = c.detach()

				if i % skip == 0:
					past_states.append(h)
				norms = []
				
				for j in range(len(past_states)):
					time_diff = torch.ones(batch_size, 1).to(device) * ((i - skip * j) / (1.0 * sq_len))
					inp = torch.cat((past_states[j].detach(), time_diff), dim=1)
					pred_mean, pred_var = predictor(inp)
					pred_mean, pred_var = pred_mean.detach(), pred_var.detach()
					norm = torch.norm(pred_var, dim=1)
					norms.append(norm)

				if len(norms) != 0:
					norms = torch.stack(norms)
					top1 = torch.topk(norms, 1, dim=0)[1]
					best_inp = []
					best_td = []
					for j in range(batch_size):
						best_inp.append(past_states[top1[0][j].item()][j])
						best_td.append([i - skip * top1[0][j].item()])

					best_td = tensor(best_td).to(device)
					best_inp = torch.stack(best_inp)
					best_inp_var = torch.cat((best_inp, best_td), dim=1)
					#print(best_id)
					pred_mean, pred_var = predictor(best_inp_var)
					#pred_var = pred_var.detach()
					# look into different ways to combine past into present
					alpha = torch.min(0.5 + pred_var, torch.ones(batch_size, hid_size).to(device))
					#h = alpha * h + (1 - alpha) * pred_mean
					h = alpha * h + (1 - alpha) * best_inp

				output, (h, c) = model(inp_x[i], (h,c))
				loss += criterion(output, inp_y[i].squeeze(1))


			loss /= (sq_len * 1.0)
			#loss = criterion(output, inp_y)
			optimizer_1.zero_grad()
			loss.backward(retain_graph=True)
			nn.utils.clip_grad_norm_(model.parameters(), 5)
			optimizer_1.step()

			arr_x = []
			arr_y = []
			p_loss = 0
			tot = 0
			inp_p = []
			out_p = []
			for i in range(len(past_states)):
				for j in range(i):
					time_diff = torch.ones(batch_size, 1).to(device) * ((skip * (i - j)) / (1.0 * sq_len))
					inp = torch.cat((past_states[j], time_diff), dim=1)
					inp_p.append(inp.detach())
					out_p.append(past_states[i].detach())
			inp_p = torch.stack(inp_p)
			out_p = torch.stack(out_p)
			pred_mean, pred_var = predictor(inp_p)
			#print(pred_var)
			pred_loss = torch.log(2 * math.pi * pred_var) + torch.pow((pred_mean - out_p) / (pred_var + 1e-7), 2)
			p_loss += pred_loss.mean()
			tot += 1

			p_loss /= tot
			optimizer_2.zero_grad()
			p_loss.backward()
			nn.utils.clip_grad_norm_(predictor.parameters(), 5)
			optimizer_2.step()
			
			loss_val = loss.item()
			p_val = np.exp(p_loss.item())
			print(z+1, loss_val, p_val)
			writer.add_scalar('/model200loss', loss_val, ctr)
			#writer.add_scalar('/predloss', p_val, ctr)
			ctr += 1
			#print(output, inp_y)
		t_loss, accuracy = test_model(model, test_x, test_y, criterion)
		best_acc = max(best_acc, accuracy)
		print('best accuracy ' + str(best_acc))
		writer.add_scalar('model200acc', accuracy, epoch)

device = torch.device('cuda')

net = Net(inp_size, hid_size, out_size).to(device)
predictor = Predictor(hid_size, 150).to(device)
criterion = nn.CrossEntropyLoss()
optimizer_1 = optim.Adam(list(net.parameters()) + list(predictor.parameters()), lr=0.001)
optimizer_2 = optim.Adam(predictor.parameters(), lr=0.001)
train_model(net, n_epochs, criterion, optimizer_1, optimizer_2)
writer.close()