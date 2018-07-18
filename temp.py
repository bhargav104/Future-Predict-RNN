import numpy as np
import sys

f = open('ptb/ptb.train.txt', 'r')
words = {}
mxlen = 0
for x in f:
	x = x.strip().split()
	mxlen = max(mxlen, len(x))
	for y in x:
		words[y] = 1

print(mxlen)
