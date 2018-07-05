#import cPickle as pkl
import gzip
import numpy
import os
import sys

_data_cache = dict()

def get_data(which_set, path):
    if which_set not in _data_cache:
        data = numpy.load(path)
    return data[which_set]


class PTBTextIterator:
    """Simple Bitext iterator."""
    def __init__(self,
                 which_set,
                 length=100,
                 batch_size=128,
                 num_batch=None,
                 which_data='ptb',
                 path=None,
                 shuffle_every_epoch=False):

        self.batch_size = batch_size
        self.shuffle_every_epoch = shuffle_every_epoch

        self.which_set = which_set
        self.length = length
        if path is not None:
            if which_data == 'ptb':
                path = os.path.join(path, 'char_penntree.npz')
            elif which_data == 'wiki':
                path = os.environ["CHAR_LEVEL_WIKI_NPZ"]
                if which_set == 'train':
                    which_set = 'train_chars'
                elif which_set == 'valid':
                    which_set = 'valid_chars'
                elif which_set == 'test':
                    which_set = 'test_chars'
            elif which_data == 'text8':
                path = os.path.join(path, 'text8_2.npz')
        
        self.data = get_data(which_set, path)
        self.size = int(len(self.data) / self.batch_size) * self.batch_size
        self.data2 = self.chop()
        if num_batch is None:
            num_batch = numpy.iinfo(numpy.int64).max
        self.num_batch = num_batch
        self.idx = 0

    def chop(self):
        # Reshape to non-overlapping examples
        if self.shuffle_every_epoch:
            roll_step = numpy.random.randint(len(self.data))
            data = numpy.roll(self.data, axis=0, shift=roll_step)
        else:
            data = self.data
        batch_data = data[:self.size].reshape(self.batch_size, -1).transpose()
        # batch_size * sequence_length

        return batch_data # shape=(sequence_temporal_direction, batch_size)

    def __iter__(self):
        return self

    def next(self):

        if self.idx * (self.length - 1) >= min(self.data2.shape[0], self.num_batch * self.length):
            self.idx = 0
            self.data2 = self.chop()
            raise StopIteration
        else:
            # It is important to make one symbol to overlap!
            # If not, you are basically skipping a symbol.
            batch = self.data2[self.idx * (self.length - 1):self.idx * (self.length - 1) + self.length]
            batch = batch.astype('int64')
            batch = numpy.transpose(batch)
            self.idx += 1

        return batch


which_data = 'text8'
'''
train = PTBTextIterator(which_data=which_data, which_set='train',
                       batch_size=128,
                       shuffle_every_epoch=1,
                       length=500, path='./text8') # 399, 3928, 158, 1409

valid = PTBTextIterator(which_data=which_data, which_set='valid',
                       batch_size=128,
                       shuffle_every_epoch=0,
                       length=500, path='./text8') # 32 , 218, 12, 78

test = PTBTextIterator(which_data=which_data, which_set='test',
                       batch_size=128,
                       shuffle_every_epoch=0,
                       length=500, path='./text8') # 32 , 218, 13, 78
'''