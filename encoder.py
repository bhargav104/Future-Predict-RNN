import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lstm_cell import LSTM
from tensorboardX import SummaryWriter
import argparse
parser = argparse.ArgumentParser(description='auglang parameters')
parser.add_argument('--full', action='store_true', default=False, help='Use full BPTT')
parser.add_argument('--trunc', type=int, default=5, help='size of H truncations')
parser.add_argument('--p-detach', type=float, default=1.0, help='probability of detaching each timestep')

args = parser.parse_args()

np.random.seed(400)
torch.manual_seed(400)
torch.cuda.manual_seed(400)

writer = SummaryWriter()

word_dict = {}
f = open('ptb/ptb.train.txt', 'r')
T = 100
n_batches = 50000
batch_size = 100
lstm_size = 1000
vocab_size = 10001
emd_size = 1000
lr = 0.0001
test_fq = 300

for x in f:
	x = x.strip().split()
	for y in x:
		word_dict[y] = 0
f = open('ptb/ptb.train.txt', 'r')
ctr = 1
for x in f:
	x = x.strip().split()
	for y in x:
		if word_dict[y] == 0:
			word_dict[y] = ctr
			ctr += 1

def batch_generator(file, batch_size):
	lens = torch.zeros(batch_size, T)
	ctr = 0
	batch = []
	while True:
		f = open(file, 'r')
		for x in f:
			x = x.strip().split()
			arr = np.zeros(T)
			arr[0] = vocab_size - 1
			lens[ctr, :(len(x)+2)] = 1
			for i in range(len(x)):
				arr[i + 1] = word_dict[x[i]]
			ctr += 1
			batch.append(arr)
			if ctr == batch_size:
				batch = torch.LongTensor(batch)
				yield batch, lens
				batch = []
				ctr = 0
				lens = torch.zeros(batch_size, T)

class Encoder(nn.Module):

	def __init__(self, vocab_size, emd_size, hid_size):
		super().__init__()
		self.embed = nn.Embedding(vocab_size, emd_size)
		self.lstm = LSTM(emd_size, hid_size)

	def forward(self, x, state):
		x = self.embed(x)
		x, (h, c) = self.lstm(x, state)
		return x, (h, c)

class Decoder(nn.Module):

	def __init__(self, vocab_size, emd_size, hid_size):
		super().__init__()
		self.embed = nn.Embedding(vocab_size, emd_size)
		self.lstm = LSTM(emd_size, hid_size)
		self.fc1 = nn.Linear(hid_size, vocab_size)

	def forward(self, x, state):
		x = self.embed(x)
		x, (h, c) = self.lstm(x, state)
		x = self.fc1(x)
		return x, (h, c)

def test_model(encoder, decoder, file, criterion):

	test = batch_generator(file,batch_size)
	n_test = 34
	with torch.no_grad():
		test_loss = 0
		for z in range(n_test):
			batch, lens = next(test)
			batch, lens = batch.to(device), lens.to(device)
			sizes = torch.sum(lens, dim=1)
			batch = batch.t()
			lens = lens.t()
			h = torch.zeros(batch_size, lstm_size).to(device)
			c = torch.zeros(batch_size, lstm_size).to(device)
		
			encodings = []
			for i in range(T):
				output, (h, c) = encoder(batch[i], (h, c))
				encodings.append((h, c))
		
			d_h = []
			d_c = []
			for i in range(batch_size):
				d_h.append(encodings[int(sizes[i].item()) - 1][0][i])
				d_c.append(encodings[int(sizes[i].item()) - 1][1][i])
		
			d_h = torch.stack(d_h, dim=0)
			d_c = torch.stack(d_c, dim=0)
			#d_c = torch.zeros(batch_size, lstm_size).to(device)
			inp = torch.ones(batch_size, dtype=torch.long).to(device) * (vocab_size - 1)
			loss = 0
			for i in range(T):
				output, (d_h, d_c) = decoder(inp , (d_h, d_c))
				inp = torch.max(output.detach(), dim=1)[1]
				loss += (criterion(output, batch[i]) * lens[i]).sum()
		
			loss /= sizes.sum()
			test_loss += loss

	test_loss /= n_test
	test_loss = test_loss.item()
	print('test loss ', test_loss)
	return test_loss

def train_model(encoder, decoder):
	
	train = batch_generator('ptb/ptb.train.txt', batch_size)
	criterion = nn.CrossEntropyLoss(reduce=False)
	optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
	t_ctr = 0
	for z in range(n_batches):
		batch, lens = next(train)
		batch, lens = batch.to(device), lens.to(device)
		sizes = torch.sum(lens, dim=1)
		batch = batch.t()
		lens = lens.t()
		h = torch.zeros(batch_size, lstm_size).to(device)
		c = torch.zeros(batch_size, lstm_size).to(device)
		
		encodings = []
		for i in range(T):
			if not args.full:
				val = np.random.random(size=1)[0]
				if val <= args.p_detach:
					h = h.detach()
			output, (h, c) = encoder(batch[i], (h, c))
			encodings.append((h, c))
		
		d_h = []
		d_c = []
		for i in range(batch_size):
			d_h.append(encodings[int(sizes[i].item()) - 1][0][i])
			d_c.append(encodings[int(sizes[i].item()) - 1][1][i])
		
		d_h = torch.stack(d_h, dim=0)
		d_c = torch.stack(d_c, dim=0)
		#d_c = torch.zeros(batch_size, lstm_size).to(device)
		inp = torch.ones(batch_size, dtype=torch.long).to(device) * (vocab_size - 1)
		loss = 0
		for i in range(T):
			if not args.full:
				val = np.random.random(size=1)[0]
				if val <= args.p_detach:
					h = h.detach()
			output, (d_h, d_c) = decoder(inp , (d_h, d_c))
			inp = torch.max(output.detach(), dim=1)[1]
			loss += (criterion(output, batch[i]) * lens[i]).sum()
		
		loss /= sizes.sum()
		print(loss.item())
		writer.add_scalar('/encoderTestTrain', loss, z)
		optimizer.zero_grad()
		loss.backward()
		norms = nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
		optimizer.step()
		if z % test_fq == test_fq - 1:
			print(z)
			test_loss = test_model(encoder, decoder, 'ptb/ptb.valid.txt', criterion)
			writer.add_scalar('/encoderTestValid', test_loss, t_ctr)
			t_ctr += 1

device = torch.device('cuda')
encoder = Encoder(vocab_size, emd_size, lstm_size).to(device)
decoder = Decoder(vocab_size, emd_size, lstm_size).to(device)
train_model(encoder, decoder)

'''
EncoderTest - with cell, no cell. lr = 0.001, 0.0001. p-detach = 0.9, 0.75, 0.5, 0.25, 0.1, 0.4, 0.05
lr = 0.001 best = full - 3.341, p-detach=0.5 - 3.268
lr = 0.0001 best = full - 3.208, p-detach=0.9 - 3.023
'''