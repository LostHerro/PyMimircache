import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from PyMimircache.cache.abstractCache import Cache
from PyMimircache.cacheReader.requestItem import Req
from sklearn.preprocessing import normalize
from collections import OrderedDict

'''
Defining the Neural Netowork
'''

# To be implemented
def avg_fut_rd(vtime):
    return random.random()


# TODO: Implement the vtime of the request; probably use the head of each feature vector?
class CacheNet(nn.Module):

    N_TRUE_FEATURES = 17

    def __init__(self, p=0.0):
        super(CacheNet, self).__init__()
        self.in_layer = nn.Linear(self.N_TRUE_FEATURES, 64)
        self.in_drop = nn.Dropout(p=p)
        #self.h1_layer = nn.Linear(1024, 256)
        #self.h1_drop = nn.Dropout(p=p)
        #self.h2_layer = nn.Linear(256, 64)
        #self.h2_drop = nn.Dropout(p=p)
        self.h3_layer = nn.Linear(64,16)
        self.h3_drop = nn.Dropout(p=p)
        self.h4_layer = nn.Linear(16,4)
        self.h4_drop = nn.Dropout(p=p)
        self.out_layer = nn.Linear(4, 1)

    # Head of feature vector is the virtual time (column 0)
    def forward(self, inputs):
        vtimes = inputs[:, 0].numpy()
        avg_rds = torch.Tensor([[avg_fut_rd(t)] for t in vtimes])
        inputs = inputs[:, 1:]
        inputs = F.relu(self.in_layer(inputs))
        inputs = self.in_drop(inputs)
        #inputs = F.relu(self.h1_layer(inputs))
        #inputs = self.h1_drop(inputs)
        #inputs = F.relu(self.h2_layer(inputs))
        #inputs = self.h2_drop(inputs)
        inputs = F.relu(self.h3_layer(inputs))
        inputs = self.h3_drop(inputs)
        inputs = F.relu(self.h4_layer(inputs))
        inputs = self.h4_drop(inputs)
        inputs = self.out_layer(inputs)
        output = torch.atan(inputs - avg_rds)

        return output


'''
Defining the Cache
'''

class SCache(Cache):

    def __init__(self, cache_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}
        self.counter = 0
        self.index = 0

    def evict(self, **kwargs):
        pass



