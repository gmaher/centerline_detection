import random
import itertools
import threading
import time
import logging
import random
import queue

class Dataset(object):
    def __init__(self):
        raise RuntimeError("Abstract not implemented")
    def __iter__(self):
        return self
    def __next__(self):
        raise RuntimeError("Abstract not implemented")
    def get(self):
        raise RuntimeError("Abstract not implemented")

class ListDataset(Dataset):
    def __init__(self, list_):
        self.list = list_
        self.index = -1

    def __next__(self):
        if self.index >= len(self.list)-1:
            raise StopIteration
        else:
            self.index += 1
            return self.list[self.index]

    def get(self):
        return self.__class__(self.list)

class IDListDataset(Dataset):
    def __init__(self, id_list, reader):
        self.id_list = id_list
        self.reader  = reader
        self.index = -1

    def __next__(self):
        if self.index >= len(self.id_list)-1:
            raise StopIteration
        else:
            self.index += 1
            return self.reader(self.id_list[self.index])

    def get(self):
        return self.__class__(self.id_list, self.reader)

class RandomDataset(IDListDataset):
    def __next__(self):
        id_ = random.choice(self.id_list)
        return self.reader(id_)

class MappedDataset(Dataset):
    def __init__(self, dataset, f):
        self.dataset = dataset.get()
        self.f = f
    def __next__(self):
        d = self.dataset.__next__()
        return self.f(d)
    def get(self):
        d = self.dataset.get()
        return self.__class__(d,self.f)

class BatchDataset(Dataset):
    def __init__(self,dataset,batcher, batch_size=10):
        self.dataset = dataset.get()
        self.batcher = batcher
        self.batch_size = batch_size
    def __next__(self):
        items = []
        while len(items) < self.batch_size:
            d = self.dataset.__next__()
            items.append(d)
        return self.batcher(items)
    def get(self):
        d = self.dataset.get()
        return self.__class__(d, self.batcher, self.batch_size)

class LRMappedDataset(Dataset):
    def __init__(self, dataset_l, dataset_r, f):
        self.dataset_l = dataset_l.get()
        self.dataset_r = dataset_r.get()
        self.f         = f
    def __next__(self):
        dl = self.dataset_l.__next__()
        dr = self.dataset_r.__next__()
        return self.f(dl,dr)
    def get(self):
        dl = self.dataset_l.get()
        dr = self.dataset_r.get()
        return self.__class__(dl,dr,self.f)

class FunctionDictDataset(Dataset):
    def __init__(self, datasets, functionDict):
        self.datasets = []
        for d in datasets:
            self.datasets.append(d.get())
        self.functionDict = functionDict
    def __next__(self):
        t = [d.__next__() for d in self.datasets]
        d = {}
        for k,f in self.functionDict.items(): d[k]=f(*t)
        return d
    def get(self):
        d_ = [d.get() for d in self.datasets]
        return self.__class__(d_, self.functionDict)

class MergedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = []
        for d in datasets:
            self.datasets.append(d.get())
    def __next__(self):
        return (d.__next__() for d in self.datasets)
    def get(self):
        l = []
        for d in self.datasets:
            l.append(d.get())
        return self.__class__(l)

class ReaderThread(threading.Thread):
    """Note this class is a thread, so it runs in a separate thread parallel
    to the main program"""
    def __init__(self, q, dataset, group=None, target=None, name="producer",
                 args=(), kwargs=None, verbose=None):
        super(ReaderThread,self).__init__()
        self.target    = target
        self.name      = name
        self.dataset   = dataset
        self.q         = q

    def run(self):
        while True:
            if not self.q.full():
                item_ = self.dataset.__next__()
                self.q.put(item_)
            else:
                time.sleep(random.random())
        return

class ThreadDataset(object):
    def __init__(self, dataset, queue_size=200, num_threads=100):
        self.queue_size   = queue_size
        self.q            = queue.Queue(queue_size)
        self.dataset      = dataset.get()
        self.num_threads  = num_threads

        self.readers = []
        for i in range(self.num_threads):
            t = ReaderThread(self.q, self.dataset, name='producer'+str(i))
            t.setDaemon(True)
            t.start()
            self.readers.append(t)

    def __next__(self):
        while True:
            try:
                item = self.q.get()
                return item
            except:
                time.sleep(random.random())

    def get(self):
        return self.__class__(self.dataset.get(), self.queue_size, self.num_threads)
