import theano
import numpy as np


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
            return (self.num_iter - 1), self.sample(**params)
        else:
            raise StopIteration()

    def sample_params(self):
        raise NotImplementedError()

    def sample(self):
        raise NotImplementedError()


class CopyTask(Task):

    def __init__(self, size, max_length, min_length=1, max_iter=None, \
        batch_size=1, end_marker=False):
        super(CopyTask, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.size = size
        self.min_length = min_length
        self.max_length = max_length
        self.end_marker = end_marker

    def sample_params(self, length=None):
        if length is None:
            length = np.random.randint(self.min_length, self.max_length + 1)
        return {'length': length}

    def sample(self, length):
        sequence = np.random.binomial(1, 0.5, (self.batch_size, length, self.size))
        example_input = np.zeros((self.batch_size, 2 * length + 1 + self.end_marker, \
            self.size + 1), dtype=theano.config.floatX)
        example_output = np.zeros((self.batch_size, 2 * length + 1 + self.end_marker, \
            self.size + 1), dtype=theano.config.floatX)

        example_input[:, :length, :self.size] = sequence
        example_input[:, length, -1] = 1
        example_output[:, length + 1:2 * length + 1, :self.size] = sequence
        if self.end_marker:
            example_output[:, -1, -1] = 1

        return example_input, example_output


class RepeatCopyTask(Task):

    def __init__(self, size, max_length, max_repeats=20, min_length=1, \
        min_repeats=1, unary=False, max_iter=None, batch_size=1, end_marker=False):
        super(RepeatCopyTask, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.size = size
        self.min_length = min_length
        self.max_length = max_length
        self.min_repeats = min_repeats
        self.max_repeats = max_repeats
        self.unary = unary
        self.end_marker = end_marker

    def sample_params(self, length=None, repeats=None):
        if length is None:
            length = np.random.randint(self.min_length, self.max_length + 1)
        if repeats is None:
            repeats = np.random.randint(self.min_repeats, self.max_repeats + 1)
        return {'length': length, 'repeats': repeats}

    def sample(self, length, repeats):
        sequence = np.random.binomial(1, 0.5, (self.batch_size, length, self.size))
        num_repeats_length = repeats if self.unary else 1
        example_input = np.zeros((self.batch_size, (repeats + 1) * length + \
            num_repeats_length + 1 + self.end_marker, self.size + 2), dtype=theano.config.floatX)
        example_output = np.zeros((self.batch_size, (repeats + 1) * length + \
            num_repeats_length + 1 + self.end_marker, self.size + 2), dtype=theano.config.floatX)

        example_input[:, :length, :self.size] = sequence
        for j in range(repeats):
            example_output[:, (j + 1) * length + num_repeats_length + 1:\
            (j + 2) * length + num_repeats_length + 1, :self.size] = sequence
        if self.unary:
            example_input[:, length:length + repeats, -2] = 1
        else:
            example_input[:, length, -2] = repeats / float(self.max_repeats)
        example_input[:, length + num_repeats_length, -1] = 1
        if self.end_marker:
            example_output[:, -1, -1] = 1

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

        items = np.random.binomial(1, 0.5, (self.batch_size, item_length, self.size, num_items))
        queries = np.random.randint(num_items - 1, size=self.batch_size)
        example_input = np.zeros((self.batch_size, (item_length + 1) * (num_items + 2), \
            self.size + 2), dtype=theano.config.floatX)
        example_output = np.zeros((self.batch_size, (item_length + 1) * (num_items + 2), \
            self.size + 2), dtype=theano.config.floatX)

        for j in range(num_items):
            example_input[:, j * (item_length + 1), -2] = 1
            example_input[:, item_slice(j), :self.size] = items[:,:,:,j]
        example_input[:, num_items * (item_length + 1), -1] = 1
        for batch in range(self.batch_size):
            example_input[batch, item_slice(num_items), :self.size] = items[batch,:,:,queries[batch]]
            example_output[batch, -item_length:, :self.size] = items[batch,:,:,queries[batch] + 1]
        example_input[:, (num_items + 1) * (item_length + 1), -1] = 1

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

    def sample(self, length):
        sequence = np.zeros((self.batch_size, length + 1, 1), dtype=theano.config.floatX)
        head = np.random.binomial(1, 0.5, (self.batch_size, self.ngrams))
        sequence[:, :self.ngrams, 0] = head
        index = np.dot(head, 1 << (np.arange(self.ngrams, 0, -1) - 1))
        mask = (1 << (self.ngrams - 1)) - 1

        for j in range(self.ngrams, length + 1):
            b = np.random.binomial(1, self.table[index])
            sequence[:, j, 0] = b
            index = ((index & mask) << 1) + b

        return sequence[:,:-1], sequence[:,1:]


class DyckWordsTask(Task):

    def __init__(self, max_length, min_length=1, max_iter=None, batch_size=1):
        super(DyckWordsTask, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.max_length = max_length
        self.min_length = min_length

    def sample_params(self, length=None):
        if length is None:
            length = np.random.randint(self.min_length, self.max_length + 1)
        return {'length': length}

    def sample(self, length):
        example_input = np.zeros((self.batch_size, 2 * length, 1), \
            dtype=theano.config.floatX)
        example_output = np.zeros((self.batch_size, 2 * length, 1), \
            dtype=theano.config.floatX)
        is_dyck_word = np.random.binomial(1, 0.5, self.batch_size)\
                       .astype(dtype=theano.config.floatX)

        for batch in range(self.batch_size):
            if is_dyck_word[batch]:
                word = self.get_random_dyck(length)
            else:
                word = self.get_random_non_dyck(length)
            example_input[batch, :, 0] = word
            example_output[batch, :, 0] = self.get_dyck_prefix(word)

        return example_input, example_output

    def get_dyck_prefix(self, word):
        def dyck_prefixes(prefixes_and_stack, u):
            prefixes, is_valid, stack = prefixes_and_stack
            if u: stack -= 1
            else: stack += 1
            if stack < 0:
                is_valid = False
            prefixes.append(is_valid and (stack == 0))
            return (prefixes, is_valid, stack)

        prefixes, _, _ = reduce(dyck_prefixes, word, ([], True, 0))
        return prefixes

    def get_random_dyck(self, n):
        """
        Return a random Dyck word of a given semilength `n`

        This algorithm is based on a conjugacy property between words in
        the language `L = S(u^n d^{n+1})` and *Dyck words* of length 2n,
        where `S` is the group of permutations.
        This 1-to-(2n+1) correspondance between these words is given by
        the cycle lemma:

        **Cycle Lemma**: Let `A = {u, d}` be a binary alphabet and `delta`
        a "height" function such that `delta(u) = +1` and `delta(d) = -1`.
        For any word `w` in `A^*` such that `delta(w) = -1`, there exists
        a unique factorization `w = w_1 w_2` satisfying
            - `w_1` is not empty;
            - `w_2 w_1` has the Lukasiewicz property, i.e. any strict left
            factor of `w_2 w_1` satisfies `delta(v) >= 0`.
        where we extend the definition of `delta` to words by summing the
        heights of every individual letter.

        To summarize, here is the pseudo-code for this algorithm:
            - Pick a random word `w` in the language `L = S(u^n d^{n+1})`
            - Apply the cycle lemma to find the unique conjugate of
            `w` having the Lukasiewicz property
            - Return its prefix of length 2n, which is a Dyck word

        See: [Fla09], Notes I.47 and I.49 (pp.75-77)

        [Fla09] Analytic Combinatorics, *Philippe Flajolet, Robert Sedgewick*
                <http://algo.inria.fr/flajolet/Publications/AnaCombi/anacombi.html>
        """
        # Get a random element in L = u^n d^{n+1}
        w = [0] * n + [1] * (n + 1)
        np.random.shuffle(w)

        # Get the unique conjugate of w having the Lukasiewicz property
        # (Cycle Lemma)
        min_height = (0, 0)
        stack = 0
        for i in range(2 * n):
            if w[i]: stack -= 1
            else: stack += 1
            if stack < min_height[1]:
                min_height = (i + 1, stack)
        min_idx = min_height[0]
        luka = w[min_idx:] + w[:min_idx]

        return luka[:-1]

    def get_random_non_dyck(self, n):
        """
        Return a random balanced non-Dyck word of semilength `n`

        The algorithm is based on the bijection between words in the
        language `L = S(u^{n-1} d^{n+1})` and the balanced words of length
        2n that are not Dyck words. This transformation is given by the
        reflection of the letters after the first letter that violates
        the Dyck property (i.e. the first right parenthesis that does
        not have a matching left counterpart). The reflexion transformation
        is defined by transforming any left parenthesis in a right one
        and vice-versa.

        To summarize, here is the pseudo-code for this algorithm:
            - Pick a random word `w` in the language `L = S(u^{n-1} d^{n+1})`
            - Find the first letter violating the Dyck property
            - Apply the reflection transformation to the following letters
        """
        w = [0] * (n - 1) + [1] * (n + 1)
        np.random.shuffle(w)

        stack, reflection = (0, False)
        for i in range(2 * n):
            if reflection:
                w[i] = 1 * (not w[i])
            else:
                if w[i]: stack -= 1
                else: stack += 1
                reflection = (stack < 0)
        return w
