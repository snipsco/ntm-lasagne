import theano
import theano.tensor as T
import numpy as np
import time

class Task(object):

    def __init__(self, max_iter=None, batch_size=1):
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.num_iter = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if (self.max_iter is None) or (self.num_iter < self.max_iter):
            self.num_iter += 1
            params = self.sample_params()
            return self.num_iter, self.sample(**params)
        else:
            raise StopIteration()

    def sample_params(self):
        raise NotImplementedError()

    def sample(self):
        raise NotImplementedError()


class CopyTask(Task):

    def __init__(self, size, max_length, min_length=1, max_iter=None, batch_size=1):
        super(CopyTask, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.size = size
        self.min_length = min_length
        self.max_length = max_length

    def sample_params(self, length=None):
        if length is None:
            length = np.random.randint(self.min_length, self.max_length + 1)
        return {'length': length}

    def sample(self, length):
        sequence = np.random.binomial(1, 0.5, (self.batch_size, length, self.size))
        example_input = np.zeros((self.batch_size, 2 * length + 1, self.size + 1), \
            dtype=theano.config.floatX)
        example_output = np.zeros((self.batch_size, 2 * length + 1, self.size + 1), \
            dtype=theano.config.floatX)

        example_input[:, :length, :self.size] = sequence
        example_output[:, length + 1:, :self.size] = sequence
        example_input[:, length, -1] = 1

        return example_input, example_output


class RepeatCopyTask(Task):

    def __init__(self, size, max_length, max_repeats=20, min_length=1, \
        min_repeats=1, unary=False, max_iter=None, batch_size=1):
        super(RepeatCopyTask, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.size = size
        self.min_length = min_length
        self.max_length = max_length
        self.min_repeats = min_repeats
        self.max_repeats = max_repeats
        self.unary = unary

    def sample_params(self, length=None, repeats=None):
        if length is None:
            length = np.random.randint(self.min_length, self.max_length + 1)
        if repeats is None:
            repeats = np.random.randint(self.min_repeats, self.max_repeats + 1)
        return {'length': length, 'repeats': repeats}

    def sample(self, length, repeats):
        sequence = np.random.binomial(1, 0.5, (length, self.size))
        num_repeats_length = repeats if self.unary else 1
        example_input = np.zeros((1, (repeats + 1) * length + \
            num_repeats_length + 1, self.size + 2), dtype=theano.config.floatX)
        example_output = np.zeros((1, (repeats + 1) * length + \
            num_repeats_length + 1, self.size + 2), dtype=theano.config.floatX)

        example_input[0, :length, :self.size] = sequence
        for j in range(repeats):
            example_output[0, (j + 1) * length + num_repeats_length + 1:\
            (j + 2) * length + num_repeats_length + 1, :self.size] = sequence
        if self.unary:
            example_input[0, length:length + repeats, -2] = 1
        else:
            example_input[0, length, -2] = repeats / float(self.max_repeats)
        example_input[0, length + num_repeats_length, -1] = 1

        return example_input, example_output


class AssociativeRecallTask(Task):

    def __init__(self, size, max_item_length, max_num_items, \
        min_item_length=1, min_num_items=2, max_iter=None, batch_size=1):
        super(AssociativeRecallTask, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.size = size
        self.max_item_length = max_item_length
        self.max_num_items = max_num_items
        self.min_item_length = min_item_length
        self.min_num_items = min_num_items

    def sample_params(self, item_length=None, num_items=None):
        if item_length is None:
            item_length = np.random.randint(self.min_item_length, \
                self.max_item_length + 1)
        if num_items is None:
            num_items = np.random.randint(self.min_num_items, \
                self.max_num_items + 1)
        return {'item_length': item_length, 'num_items': num_items}

    def sample(self, item_length, num_items):
        def item_slice(j):
            slice_idx = j * (item_length + 1) + 1
            return slice(slice_idx, slice_idx + item_length)

        items = np.random.binomial(1, 0.5, (item_length, self.size, num_items))
        query = np.random.randint(num_items - 1)
        example_input = np.zeros((1, (item_length + 1) * (num_items + 2), \
            self.size + 2), dtype=theano.config.floatX)
        example_output = np.zeros((1, (item_length + 1) * (num_items + 2), \
            self.size + 2), dtype=theano.config.floatX)

        for j in range(num_items):
            example_input[0, j * (item_length + 1), -2] = 1
            example_input[0, item_slice(j), :self.size] = items[:,:,j]
        example_input[0, num_items * (item_length + 1), -1] = 1
        example_input[0, item_slice(num_items), :self.size] = items[:,:,query]
        example_input[0, (num_items + 1) * (item_length + 1), -1] = 1
        example_output[0, -item_length:, :self.size] = items[:,:,query + 1]

        return example_input, example_output


class DynamicNGramsTask(Task):

    def __init__(self, ngrams, max_length, min_length=1, max_iter=None, \
        table=None, batch_size=1):
        super(DynamicNGramsTask, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.ngrams = ngrams
        if table is None:
            table = self.make_table()
        self.table = table
        self.max_length = max(ngrams, max_length)
        self.min_length = min_length
        
    def make_table(self):
        return np.random.beta(0.5, 0.5, 1 << self.ngrams)

    def sample_params(self, length=None):
        if length is None:
            length = np.random.randint(self.min_length, self.max_length + 1)
        return {'length': length}

    def sample(self, length=None):
        sequence = np.zeros((1, length + 1, 1), dtype=theano.config.floatX)
        head = np.random.binomial(1, 0.5, self.ngrams)
        sequence[0, :self.ngrams, 0] = head
        index = np.dot(head, 1 << (np.arange(self.ngrams, 0, -1) - 1))
        mask = (1 << (self.ngrams - 1)) - 1

        for j in range(self.ngrams, length + 1):
            b = np.random.binomial(1, self.table[index])
            sequence[0, j, 0] = b
            index = ((index & mask) << 1) + b

        return sequence[:,:-1], sequence[:,1:]
