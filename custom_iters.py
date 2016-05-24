"""
DataAndWeightIter
    a) Combine a data iterator (e.g., ImageRecordIter) and weight iterator (e.g., CSVIter)
    b) Use case: instance-weighted softmax loss for multi-class problems
"""

import sys
import threading

import mxnet as mx


class DataAndWeightIter(mx.io.DataIter):
    """Combine data and weight iters, based on `PrefetchingIter`

    Data and label are from the first iter, while weight is from the second
    """
    def __init__(self, iters):
        super(DataAndWeightIter, self).__init__()
        if not isinstance(iters, list):
            iters = [iters]
        self.n_iter = len(iters)
        assert self.n_iter > 0
        self.iters = iters

        # set provide_data and provide_label
        self.provide_data = [('data', iters[0].provide_data[0][1]), ('weight', iters[1].provide_data[0][1])]
        self.provide_label = iters[0].provide_label

        self.batch_size = self.provide_data[0][1][0]
        self.data_ready = [threading.Event() for i in range(self.n_iter)]
        self.data_taken = [threading.Event() for i in range(self.n_iter)]
        for e in self.data_taken:
            e.set()
        self.started = True
        self.current_batch = [None for i in range(self.n_iter)]
        self.next_batch = [None for i in range(self.n_iter)]

        def prefetch_func(self, i):
            """Thread entry"""
            while True:
                self.data_taken[i].wait()
                if not self.started:
                    break
                try:
                    self.next_batch[i] = self.iters[i].next()
                except StopIteration:
                    self.next_batch[i] = None
                self.data_taken[i].clear()
                self.data_ready[i].set()

        self.prefetch_threads = [threading.Thread(target=prefetch_func, args=[self, i]) \
                                 for i in range(self.n_iter)]
        for thread in self.prefetch_threads:
            thread.setDaemon(True)
            thread.start()

    def __del__(self):
        self.started = False
        for e in self.data_taken:
            e.set()
        for thread in self.prefetch_threads:
            thread.join()

    def reset(self):
        for e in self.data_ready:
            e.wait()
        for i in self.iters:
            i.reset()
        for e in self.data_ready:
            e.clear()
        for e in self.data_taken:
            e.set()

    def iter_next(self):
        for e in self.data_ready:
            e.wait()

        if self.next_batch[0] is None:
            for i in self.next_batch:
                assert i is None, "Number of entry mismatches between iterators"
            return False
        else:
            for batch in self.next_batch:
                assert batch.pad == self.next_batch[0].pad, \
                    "Number of entry mismatches between iterators"

            self.current_batch = mx.io.DataBatch(
                # two parts: data and weight
                self.next_batch[0].data + self.next_batch[1].data,
                # only the first iter has label
                self.next_batch[0].label,
                self.next_batch[0].pad,
                self.next_batch[0].index
            )
            for e in self.data_ready:
                e.clear()
            for e in self.data_taken:
                e.set()
            return True

    def next(self):
        if self.iter_next():
            return self.current_batch
        else:
            raise StopIteration

    def getdata(self):
        return self.current_batch.data

    def getlabel(self):
        return self.current_batch.label

    def getindex(self):
        return self.current_batch.index

    def getpad(self):
        return self.current_batch.pad
