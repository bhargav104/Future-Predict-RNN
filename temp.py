import numpy as np
import sys

maxdep = 5
p_op = 0.7
ret = ''
maxlen = 30
depth = 0

for i in range(maxlen):
	val = np.random.random(size=1)[0]
	if ((val <= p_op and depth < maxdep) or depth == 0) and maxlen - i > depth:
		ret += '('
		depth += 1
	else:
		ret += ')'
		depth -= 1

print(ret)

	