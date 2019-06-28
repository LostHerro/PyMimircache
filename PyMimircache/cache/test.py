from PyMimircache.cache.abstractCache import Cache
from PyMimircache.cacheReader.requestItem import Req
from sklearn.ensemble import RandomForestRegressor
import numpy as np # do I need this?
import math
import random
import threading

'''
Class for testing minimizing random forest regression on reuse distance as
a cache replacement policy
'''
class Test(Cache):

    '''
    Initialization function: creates the real cache and a ghost cache for training purposes

    :param cache_size: the size of the cache
    :param factor: the factor (in [0,1]) scale to determine the size of the training
                   period relative to the cache size
    '''
    def __init__(self, cache_size, factor=0.5, **kwargs):
        super().__init__(cache_size, **kwargs)

        # List of previous request ids of maximum length train_period
        # Cleared at end of every training period
        self.ghost_cache = []

        # Dictionary of request id -> request item
        self.cache = {}

        # Train_period describes how many requests are given between each training process
        self.train_period = math.floor(cache_size * factor)

    '''
    Finds the reuse distance for a given request
    If there is no request in the future, returns the training period as arbitrary high value

    :param ind: the index in the ghost cache
    '''
    def reuse_distance(self, ind):
        req_id = self.ghost_cache[ind]
        try:
            return ind - self.ghost_cache.index(req_id, ind + 1)
        except ValueError:
            return self.train_period

    '''
    Trains the random forest on the ghost cache
    Stores the regressor as self.reg
    Possibly introduce random sampling for quicker training later?
    '''
    def train(self):
        train_target = []
        train_data = [[]]
        for ind in enumerate(self.ghost_cache):
            # Condition on being able to retrieve data from cache
            if (self.ghost_cache[ind] in self.cache):
                # Get reuse distance for the index
                train_target.append(self.reuse_distance(ind))
                # Get info for request as training data
                req = self.cache[self.ghost_cache[ind]]
                train_data.append([str(req.item_id()), req.size(), 
                    str(req.op()), req.cost()])
        
        # Fit the regressor
        self.reg = RandomForestRegressor()
        self.reg.fit(train_data, train_target)

        # Reset the ghost_cache to empty
        self.ghost_cache = []

    '''
    Returns whether the cache has the request id provided
    '''
    def has(self, req_id, **kwargs):
        return req_id in self.cache

    '''
    Updates all regarding the ghost_cache
    '''
    def _update(self, req_item, **kwargs):
        if (len(self.ghost_cache) >= self.train_period):
            # Create background thread to run training process
            train_thread = threading.Thread(target=self.train)
            train_thread.start()

        # Add the request to the ghost cache
        self.ghost_cache.append(req_item)
    
    '''
    Inserts the desired itme into the cache
    '''
    def _insert(self, req_item, **kwargs):
        self.cache[req_item.item_id()] = req_item

    '''
    Evicts the request with the provided id from the cache
    '''
    def evict(self, req_id, **kwargs):
        del self.cache[req_id]
    
    '''
    Eviction process for the cache. Uses the regressor and evicts
    multiple candiates for efficiency

    :param nsamples: the number of samples to take in the eviction process
                     should be power of 2 times nevicts and less than cache size
    :param nevcts: the number of cache items evicted, less than cache size
    '''
    def eviction_process(self, nsamples, nevicts, **kwargs):

        # Find the desired candidates to evict from random sampling and
        # pairwise comparison
        init_inds = random.sample(self.cache.keys(), k=nsamples)

        # Convert the randomly sampled ids to a 2d array with features for prediction
        init_cands = list(map(
            lambda x: (x, self.reg.predict([
                str(self.cache[x].item_id()),
                self.cache[x].size(),
                str(self.cache[x].op()),
                self.cache[x].cost()])), init_inds))

        # Variables for iteration
        curr_cands = init_cands
        curr_num = nsamples

        # Loop until the numbers are equal
        while (curr_num > nevicts):
            new_cands = []
            for i in range(curr_num/2):
                # these should be the estimated reuse dist
                # Since the original order is random, we can used a fixed order here
                r_dist1 = curr_cands[i][1] 
                r_dist2 = curr_cands[i + (curr_num/2)][1]

                # Minimize reuse distance
                if (r_dist1 < r_dist2):
                    new_cands.append(r_dist1)
                else:
                    new_cands.append(r_dist2)

            curr_cands = new_cands
            curr_num = curr_num / 2
        
        # Do evictions
        for pair in curr_cands:
            self.evict(pair[0]) # Should be the id

    '''
    Method that is first called when a cache request arrives

    :return: whether the request is in the cache (hit)
    '''
    def access(self, req_item, **kwargs):
        
        # Update the ghost cache
        self._update(req_item)

        if self.has(req_item.item_id()):
            return True
        else:
            self._insert(req_item)
            if (len(self.cache) < self.cache_size):
                self.eviction_process(32, 8) # Maybe scale these numbers to cache size?
            return False

    # The following methods were in the lru file so I'm just including them.
    # Don't really know why they are needed.
    def __contains__(self, req_item):
        return req_item in self.cache.values()

    def __len__(self):
        return len(self.cache)
    
    def __repr__(self):
        return "Test cache of size: {}, current size: {}".\
            format(self.cache_size, len(self.cache))
         