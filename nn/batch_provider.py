from scipy.sparse.csr import csr_matrix
import numpy as np

class BatchProvider:
    def __init__(self, data, batch_size):
        self.data = data
        self.no_examples = data.shape[0]
        self.batch_size = batch_size
        self.index = 0

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.no_examples:
            raise StopIteration

        start = self.index
        end = self.index + self.batch_size
        if end > self.no_examples:
            end = self.no_examples

        self.index += self.batch_size

        data = self.data[start:end]
        if type(data) == csr_matrix:
            data = np.asarray(data.todense())

        return data