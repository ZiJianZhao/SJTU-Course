import mxnet as mx
import numpy as np

from sklearn.cluster import KMeans

class SimpleBatch(object):
    def __init__(self, data_names, data,
            label_names, label):
        
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


class DummyIter(mx.io.DataIter):
    "A dummy iterator that always return the same batch, used for speed testing"
    def __init__(self, real_iter):
        super(DummyIter, self).__init__()
        self.real_iter = real_iter
        self.provide_data = real_iter.provide_data
        self.provide_label = real_iter.provide_label
        self.batch_size = real_iter.batch_size

        for batch in real_iter:
            self.the_batch = batch
            break

    def __iter__(self):
        return self

    def next(self):
        return self.the_batch

DEBUG = False

class SequenceIter(mx.io.DataIter):
    def __init__(self, data, label, batch_size = 32, training = True):
        # Initialization
        super(SequenceIter, self).__init__() 
        
        self.data = data
        self.label = label 
        self.batch_size = batch_size
        self.data_shape = (self.batch_size, self.data.shape[1])
        self.training = training
        # make a random data iteration plan
        if training:
            self.make_data_iter_plan()

        self.provide_data = [('data' , self.data_shape)]
        self.provide_label = [('label', (self.batch_size,))]
        self.reset()

    def __iter__(self):
        
        data = np.zeros(self.data_shape)
        label = np.zeros((self.batch_size, ))

        for begin in range(0, self.data.shape[0] - self.batch_size+1, self.batch_size):
            idx = self.shuffle_indices[begin:begin+self.batch_size]
            data[:] = self.data[idx]
            label[:] = self.label[idx]

            data_all = [mx.nd.array(data)]
            label_all = [mx.nd.array(label)]
            data_names = ['data']
            label_names = ['label']
            data_batch = SimpleBatch(data_names, data_all, label_names, label_all)
            
            yield data_batch

    def reset(self):
        if self.training:
            self.shuffle_indices = np.random.permutation(np.arange(len(self.label)))
        else:
            self.shuffle_indices = np.arange(len(self.label))

    def make_data_iter_plan(self):
        "make a random data iteration plan"
        np.random.seed(10)
        self.shuffle_indices = np.random.permutation(np.arange(len(self.label)))