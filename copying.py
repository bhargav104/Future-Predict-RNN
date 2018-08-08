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
import argparse
from torch.autograd import Variable
from vHv import get_vHv
import os
import glob
import tqdm 
parser = argparse.ArgumentParser(description='auglang parameters')
parser.add_argument('--full', action='store_true', default=False, help='Use full BPTT')
parser.add_argument('--trunc', type=int, default=5, help='size of H truncations')
parser.add_argument('--p-detach', type=float, default=1.0, help='probability of detaching each timestep')
parser.add_argument('--lstm-size', type=int, default=128, help='hidden size of LSTM')
parser.add_argument('--noise', type=float, default=0.0, help='make the gradient noisy')
parser.add_argument('--gHg', type=bool, default=False, help='track gHg during training')
parser.add_argument('--save-dir', type=str, default='default', help='save dir of the results')

args = parser.parse_args()
log_dir = 'exp/300ghg/'+args.save_dir + '/'

if os.path.isdir(log_dir):
	print('deleting contents of experiment directory')
	for f in glob.glob(log_dir+'*'):
		print(f)
		os.remove(f)

writer = SummaryWriter(log_dir=log_dir)

#writer = SummaryWriter()

torch.cuda.manual_seed(400)
torch.manual_seed(400)
np.random.seed(400)
tensor = torch.FloatTensor

n_epochs = 600
T = 300
batch_size = 100
inp_size = 1
hid_size = args.lstm_size
out_size = 9
lr = 0.001
train_size = 100000
test_size = 5000
update_fq = 50
ktrunc = args.trunc

def hard_update(target, source):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(param.data)

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
		self.lstm = LSTM(inp_size, hid_size)
		self.fc1 = nn.Linear(hid_size, out_size)

	def forward(self, x, state):
		x, new_state = self.lstm(x, state)
		x = self.fc1(x)
		return x, new_state

def test_model(model, test_x, test_y, criterion):
	loss = 0
	accuracy = 0
	inp_x= torch.transpose(test_x, 0,1)
	inp_y = torch.transpose(test_y, 0,1)
	h = torch.zeros(test_size, hid_size).to(device)
	c = torch.zeros(test_size, hid_size).to(device)

	with torch.no_grad():
		for i in range(T + 20):
			output, (h, c) = model(inp_x[i], (h, c))
			loss += criterion(output, inp_y[i].squeeze(1)).item()
			if i >= T + 10:
				preds = torch.argmax(output, dim=1)
				actual = inp_y[i].squeeze(1)
				correct = preds == actual
				accuracy += correct.sum().item()

	loss /= (T + 20.0)
	accuracy /= (500.0)

	print('test loss ' + str(loss) + ' accuracy ' + str(accuracy))
	return loss, accuracy

def get_flat_grads(model):
	ret = []
	for param in model.parameters():
		ret.append(param.grad.data.view(-1))
	ret = torch.cat(ret, dim=0)
	return ret

def vHv_fn(model, model_1, data, train_size, batch_size, criterion, T):
	(train_x, train_y) = data
	hard_update(model_1, model)
	update_grad = []
	for variable in model.parameters():
		update_grad.append(variable.grad.data.clone())
	direc = torch.cat([m.view(-1) for m in update_grad]).cpu().numpy()
	direc /= np.linalg.norm(direc)
	direc = Variable(torch.from_numpy(direc.astype("float32") ).cuda() )
	vHv_val = get_vHv(model_1, direc, (train_x, train_y), train_size, batch_size, args, criterion, T)
	return vHv_val

def forwardprop(model, inp_x, inp_y, h, c, p_detach):
	sq_len = T + 20
	loss = 0

	val = np.random.random(size=1)[0]
	# 0.8 0.6 0.4 0.2
	for i in range(sq_len):
		if p_detach != 1.0 and not args.full:
			rand_val = np.random.random(size=1)[0]
			if rand_val <= p_detach:
				h = h.detach()
		output, (h, c) = model(inp_x[i], (h, c))
		loss += criterion(output, inp_y[i].squeeze(1))

	loss /= (1.0 * sq_len)
	return loss

def train_model(model, model_1, epochs, criterion, optimizer):

	train_x, train_y = create_dataset(train_size, T)
	test_x, test_y = create_dataset(test_size, T)
	train_x, train_y = train_x.to(device), train_y.to(device)
	test_x, test_y = test_x.to(device), test_y.to(device)
	best_acc = 0.0
	ctr = 0
	global lr
	dc = 0
	gc = 0
	for epoch in range(epochs):
		print('epoch ' + str(epoch + 1))
		epoch_loss = 0
		gHg_list = []
		for z in tqdm.tqdm(range(train_size // batch_size), total=train_size // batch_size):
			ind = np.random.choice(train_size, batch_size)
			inp_x, inp_y = train_x[ind], train_y[ind]
			inp_x.transpose_(0, 1)
			inp_y.transpose_(0, 1)
			h = torch.zeros(batch_size, hid_size).to(device)
			c = torch.zeros(batch_size, hid_size).to(device)

			loss = forwardprop(model, inp_x, inp_y, h, c, p_detach=args.p_detach)
			model.zero_grad()
			loss.backward()
			norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)

			if args.gHg:
				if 1<=z<=5 or 498<=z<=502 or 996<=z<=1000: # #or 248<=z<=252 or 748<=z<=752 
					vHv_val = vHv_fn(model, model_1, (train_x, train_y), train_size, batch_size, criterion, T)
					gHg_list.append(vHv_val)
					print('ghg: {}'.format(vHv_val))
					writer.add_scalar('/300gHgval', vHv_val, gc)
					gc += 1

			
			#writer.add_scalar('/300norms', norm.item(), ctr)
			#new_grads = get_flat_grads(model)

			if args.noise != 0.0:
				for param in model.parameters():
					vals = param.data
					noise = torch.normal(mean=torch.zeros(vals.size()), std=torch.ones(vals.size()) * args.noise).to(device)
					param.data.copy_(vals + noise)


			optimizer.step()

			loss_val = loss.item()
			# print(z, loss_val)
			writer.add_scalar('/300ghgloss', loss_val, ctr)
			ctr += 1



		t_loss, accuracy = test_model(model, test_x, test_y, criterion)
		if accuracy > best_acc:
			best_acc = accuracy
			#torch.save(model.state_dict(), 'copy100noforget.pt')
			print('best accuracy ' + str(best_acc))
		writer.add_scalar('/acc', accuracy, epoch)

device = torch.device('cuda')
net = Net(inp_size, hid_size, out_size).to(device)
net_1 = Net(inp_size, hid_size, out_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)

train_model(net, net_1, n_epochs, criterion, optimizer)
#writer.close()

'''
h c
0 0
1 0
0 1
1 1
'''

'''
300 change - p-full = 0, 0.2, 0.4, 0.6, 0.8, 1, no forget gate
300 stoch - detach = 0.9, 0.75, 0.5, 0.25, 0.1
300 exp - lr = 0.001schedule, lr = 0.001, lr = 0.0001. ktrunc = 3,5,10
300 size - full, p-detach = 0.25. 64, 128, 256, 512
300 full - full data - full, 0.5. batch - full, 0.5
300 noise - full, p-detach=0.5, 0.25. noise = 0.001, 0.0025, 0.0005, 0.0001, 0.005
300 direcloss - full, 0.5, new 0.5, half-half, new 0.5
'''
