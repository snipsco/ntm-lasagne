import numpy as np

def copy(size, length):
    sequence = np.random.binomial(1, 0.5, (length, size)).astype(np.uint8)
    example_input = np.zeros((1, 2 * length + 1, size + 1))
    example_output = np.zeros((1, 2 * length + 1, size + 1))

    example_input[0, :length, :size] = sequence
    example_output[0, length + 1:, :size] = sequence
    example_input[0, length, -1] = 1

    return example_input, example_output

def copy_repeat(size, length, repeats):
    sequence = np.random.binomial(1, 0.5, (length, size)).astype(np.uint8)
    example_input = np.zeros((1, (repeats + 1) * length + 2, size + 2))
    example_output = np.zeros((1, (repeats + 1) * length + 2, size + 2))

    example_input[0, :length, :size] = sequence
    for j in range(repeats):
        example_output[0, (j + 1) * length + 2:\
        (j + 2) * length + 2, :size] = sequence
    example_input[0, length, -2] = repeats / 20.0
    example_input[0, length + 1, -1] = 1

    return example_input, example_output

def copy_repeat_alt(size, length, repeats):
    sequence = np.random.binomial(1, 0.5, (length, size)).astype(np.uint8)
    example_input = np.zeros((1, (repeats + 1) * length + repeats + 1, size + 2))
    example_output = np.zeros((1, (repeats + 1) * length + repeats + 1, size + 2))

    example_input[0, :length, :size] = sequence
    for j in range(repeats):
        example_output[0, (j + 1) * length + repeats + 1:\
        (j + 2) * length + repeats + 1, :size] = sequence
    example_input[0, length:length + repeats, -2] = 1
    example_input[0, length + repeats, -1] = 1

    return example_input, example_output


def associative_recall(size, item_length, num_items):
    items = np.random.binomial(1, 0.5, (item_length, size, num_items)).astype(np.uint8)
    query = np.random.randint(num_items - 1)
    example_input = np.zeros((1, (item_length + 1) * (num_items + 2), size + 2))
    example_output = np.zeros((1, (item_length + 1) * (num_items + 2), size + 2))
    def item_slice(j):
        return slice(j * (item_length + 1) + 1, j * (item_length + 1) + item_length + 1)

    for j in range(num_items):
        example_input[0, j * (item_length + 1), -2] = 1
        example_input[0, item_slice(j), :size] = items[:,:,j]
    example_input[0, num_items * (item_length + 1), -1] = 1
    example_input[0, item_slice(num_items), :size] = items[:,:,query]
    example_input[0, (num_items + 1) * (item_length + 1), -1] = 1
    example_output[0, -item_length:, :size] = items[:,:,query + 1]

    return example_input, example_output

def dynamic_ngrams(table, ngram, length):
    sequence = np.zeros((1, length + 1, 1))
    head = np.random.binomial(1, 0.5, ngram)
    sequence[0, :ngram, 0] = head
    index = np.dot(head, 1 << (np.arange(ngram, 0, -1) - 1))
    mask = (1 << (ngram - 1)) - 1

    for j in range(ngram, length + 1):
        b = np.random.binomial(1, table[index])
        sequence[0, j, 0] = b
        index = ((index & mask) << 1) + b

    return sequence[:,:-1], sequence[:,1:], sequence

def dynamic_ngrams_table(ngram):
    return np.random.beta(0.5, 0.5, 1 << ngram)

def dynamic_ngrams_bayesian_optimum(sequence, ngram):
    length = sequence.size
    counts = np.zeros((1 << ngram, 2))
    optimum = 0.5 * np.ones((1, length - 1, 1))
    index = int(np.dot(sequence[0, :ngram, 0], \
        1 << (np.arange(ngram, 0, -1) - 1)))
    mask = (1 << (ngram - 1)) - 1

    for j in range(ngram, length):
        b = sequence[0, j, 0]
        counts[index, b] += 1
        optimum[0, j - 1, 0] = (0.5 + counts[index, 1]) / (1 + np.sum(counts[index]))
        index = int(((index & mask) << 1) + b)

    return optimum