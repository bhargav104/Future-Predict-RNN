import os
import torch
import sys
from collections import Counter


device = torch.device('cuda')
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'ptb.train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'ptb.valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'ptb.test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split()
                for word in words:
                    for letter in word:
                        self.dictionary.add_word(letter)
                    self.dictionary.add_word(' ')
                    tokens += len(word) + 1
                self.dictionary.add_word('~')
                tokens += 1

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split()
                for word in words:
                    for letter in word:
                        ids[token] = self.dictionary.word2idx[letter]
                        token += 1
                    ids[token] = self.dictionary.word2idx[' ']
                    token += 1
                ids[token] = self.dictionary.word2idx['~']
                token += 1

        return ids

def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous().to(device)
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


corpus = Corpus('./penntree')
print(batchify(corpus.train, 100, None).size()[0])