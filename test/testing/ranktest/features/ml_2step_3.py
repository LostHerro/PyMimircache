import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random, math, time, sys

from collections import deque
from PyMimircache import Cachecow
from heapdict import heapdict

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt



'''
Initial Stuff
'''

file_name = sys.argv[1]
if sys.argv[2] == 'med':
    num = '1'
elif sys.argv[2] == 'q3':
    num = '2'
q3_fut_rd_arr = pd.read_csv('fut_rd_stats/' + file_name + '_5000.csv')[num].to_numpy()

mode = sys.argv[3] # "perfect", "break", or "none"

train_factor = random.uniform(0.6, 0.7)

med_q3 = np.median(q3_fut_rd_arr[:int(train_factor*len(q3_fut_rd_arr))])
damp_factor = med_q3 / 1.5

def avg_fut_rd(vtime, mode=mode):
    if mode == 'none':
        return med_q3
    elif mode == 'perfect':
        return q3_fut_rd_arr[int(vtime / 50)]
    elif mode == 'break':
        if int(vtime / 50) > int(train_factor*len(q3_fut_rd_arr)):
            return med_q3
        else:
            return q3_fut_rd_arr[int(vtime / 50)]
    


# TODO: Implement the vtime of the request; probably use the head of each feature vector?
class CacheNet(nn.Module):

    N_TRUE_FEATURES = 17

    def __init__(self, p=0.0):
        super(CacheNet, self).__init__()
        self.in_layer = nn.Linear(self.N_TRUE_FEATURES, 64)
        self.in_drop = nn.Dropout(p=p)
        #self.h1_layer = nn.Linear(1024,256)
        #self.h1_drop = nn.Dropout(p=p)
        #self.h2_layer = nn.Linear(64,40)
        #self.h2_drop = nn.Dropout(p=p)
        self.h3_layer = nn.Linear(64,16)
        self.h3_drop = nn.Dropout(p=p)
        self.h4_layer = nn.Linear(16,4)
        self.h4_drop = nn.Dropout(p=p)
        self.out_layer = nn.Linear(4, 1)

    # Head of feature vector is the virtual time (column 0)
    def forward(self, inputs):
        inputs = inputs[:, 1:]
        inputs = self.in_layer(inputs)
        #inputs = F.relu(self.h1_layer(inputs))
        #inputs = self.h1_drop(inputs)
        #inputs = F.relu(self.h2_layer(inputs))
        #inputs = self.h2_drop(inputs)
        inputs = F.relu(self.h3_layer(inputs))
        inputs = self.h3_drop(inputs)
        inputs = F.relu(self.h4_layer(inputs))
        inputs = self.h4_drop(inputs)
        inputs = self.out_layer(inputs)
        output = inputs

        return output


'''
Data Processing Section
'''
N_FEATURES = 20 #???? idk

def get_next_access_dist(id_ser):

    reverse_data = deque()
    next_access_dist = []
    next_access_time = {}

    for req in id_ser:
        reverse_data.appendleft(req)

    for index, req in enumerate(reverse_data):
        if req in next_access_time:
            next_access_dist.append(index - next_access_time[req])
        else:
            next_access_dist.append(len(q3_fut_rd_arr) * 50)
        next_access_time[req] = index
    
    next_access_dist.reverse()
    return next_access_dist


def create_dist_df(feature_df, samples, dists, start_time, eval=False):

    # Returns logistic virtual distance since t_now (sigmoid)
    def get_logit_dist(next_access_dist, timestamp, delta):

        if next_access_dist != -1:
            return 1/(1 + np.exp(-(next_access_dist - delta
                 - avg_fut_rd(timestamp))/damp_factor))
        else: # return 1
            return 1

    
    train_data = []
    for t in samples:
        delta = 0
        if not eval:
            delta = int(random.random() * dists[t])
        
        if delta + t < len(feature_df) + start_time:
            if not eval:
                logit_dist_val = get_logit_dist(dists[t], t, delta)
            else:
                logit_dist_val = 0 # dummy value

            ser = feature_df.loc[t].to_list()
            for i in range(3,8):
                if ser[i] == 0:
                    ser[i] = med_q3*20
            ser += [delta] + [logit_dist_val]
            train_data.append(ser)

    full_df = pd.DataFrame(data=train_data,
        columns=(list(range(N_FEATURES)) + ['final']))

    return full_df

def gen_train_eval_data(df):

    df = df.iloc[int(0.05*df.shape[0]):int(0.95*df.shape[0])].reset_index(drop=True)
    df_len = df.shape[0]
    df.insert(loc=2, column='vtime', value=df.index)
    
    tc = time.time()
    n_samples = 500000

    time_samples = np.random.randint(0, int(train_factor * df_len), size=n_samples)
    learn_data = df.iloc[:int(train_factor * df_len)]
    train_dists = get_next_access_dist(learn_data['id'])
    
    train_df = create_dist_df(learn_data, time_samples, train_dists, 0)

    td = time.time()
    print('Time to Construct Training DataFrame: ', str(td-tc))

    time_samples = list(range(int(train_factor * df_len), df_len))
    eval_data = df.iloc[int(train_factor * df_len):]
    eval_dists = get_next_access_dist(eval_data['id'])

    eval_df = create_dist_df(eval_data, time_samples, eval_dists, int(train_factor * df_len), eval=True)

    # For later purposes of PyMimircache readers
    with open('temp_trace_' + file_name + '.txt', 'w+') as f:
        for i in eval_data['id']:
            f.write(str(i))
            f.write('\n')

    te = time.time()
    print('Time to Construct Evaluation DataFrame: ', str(te-td))

    return train_df, eval_df

'''
Pytorch Integration Section
'''
t1 = time.time()

train_df, eval_df = gen_train_eval_data(pd.read_csv('ranktest/features/' + file_name + '_feat16.csv'))
#normalizing_func = lambda x: (x-np.mean(x, axis=0))/np.std(x, axis=0)
def normalizing_func(x):
    stdev = np.std(x, axis=0)
    ret = np.zeros(x.shape, dtype='float64')
    for i in range(ret.shape[1]): # iterate across columns
        if stdev[i] != 0:
            ret[:, i] = (x[:, i] - np.mean(x[:, i]))/stdev[i]
    return ret
#print(train_df)
#print(eval_df)

train_feat = train_df.drop(columns=[0,1,
    'final']).astype('float64').to_numpy()
train_target = train_df[['final']].astype('float64').to_numpy()

train_feat = np.concatenate((train_feat[:,[0]], normalizing_func(train_feat[:,1:])), axis=1)

train_feat = torch.tensor(train_feat, dtype=torch.float)
train_target = torch.tensor(train_target, dtype=torch.float)

eval_ids = eval_df[1].to_list() # For later
eval_feat = eval_df.drop(columns=[0,1,
    'final']).astype('float64').to_numpy()
#eval_feat = np.concatenate((eval_feat[:,[0]], normalizing_func(eval_feat[:,1:])), axis=1)
#eval_feat = torch.tensor(eval_feat, dtype=torch.float)


model = CacheNet(p=0.5)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

lambda1 = lambda epoch: 0.995
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
#train_feat = torch.randn(len(train_target), 34)

model.train()
print(train_target)

vtimes = train_feat[:, 0].numpy()
avg_rds = torch.Tensor([[avg_fut_rd(t)] for t in vtimes])

for t in range(300):
    # Forward Pass
    y_pred = model(train_feat)
    y_pred = torch.sigmoid((y_pred - avg_rds)/damp_factor)

    # Loss
    loss = criterion(y_pred, train_target)
    if (t % 10 == 0):
        print(t, loss.item())
    if (t % 20 == 0):
        print(y_pred)

    # Backward Pass And Update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

# Evaluation of Model
with torch.no_grad():

    model.eval()
    # Training Score
    y_pred = model(train_feat)
    y_pred = torch.sigmoid((y_pred - avg_rds)/damp_factor)
    print('Training Score:')
    print(criterion(y_pred, train_target))
    print(y_pred)  

    t2 = time.time()
    print('Time to Train: ', str(t2-t1))

    # LRU/Opt comparison setup
    c = Cachecow()
    c.open('temp_trace_' + file_name + '.txt')

    max_cache_size = 40000
    if len(sys.argv) > 4:
        max_cache_size = int(sys.argv[4])

    opt_hr_dict = c.get_hit_ratio_dict('Optimal', cache_size=max_cache_size)
    lru_hr_dict = c.get_hit_ratio_dict('LRU', cache_size=max_cache_size)
    rand_hr_dict = c.get_hit_ratio_dict('Random', cache_size=max_cache_size)

    def eviction_process(cache_dict, sample_size, n_evicts, ts):
        # Ensure that at least one element is evicted
        if sample_size == 0:
            sample_size = 1
        if n_evicts == 0:
            n_evicts = 1
        
        # Randomly sample the cache for eviction candidates
        reqs = random.sample(cache_dict.items(), sample_size)
        id_lst = [req[0] for req in reqs]
        eval_features = np.zeros((sample_size, 18), dtype='float64')

        for i, (ident, ind) in enumerate(reqs):
            eval_i = eval_feat[[ind]]
            eval_i[0, -1] = ts - cache_dict[ident] # changing the delta value
            eval_features[i] = eval_i
        
        # Run Model
        eval_features = torch.Tensor(normalizing_func(eval_features))
        eval_values = model(eval_features)
        eval_values = [eval_values[i, 0] for i in range(len(eval_values))]

        # Get indices for eviction candidates
        evict_inds = np.argsort(np.negative(eval_values))[:n_evicts]
        
        # Do deletions
        for ind in evict_inds:
            del cache_dict[id_lst[ind]]



    # Manually get hit ratios for ml model...
    """ eval_values = model(eval_feat)
    eval_values = [eval_values[i, 0] for i in range(len(eval_values))] """

    length = len(eval_ids)
    cache_size_factor = 1.5
    cache_sizes = [int(cache_size_factor**i) for i in range(int(np.log(max_cache_size)/
        np.log(cache_size_factor)) + 1)] + [max_cache_size]
    hit_ratios = []
    
    for cache_size in cache_sizes:
        cache = {}
        n_hits = 0
        print(cache_size)
        for ts, ident in enumerate(eval_ids):
            if ident in cache:
                n_hits += 1
                cache[ident] = ts # update cache
            else:
                cache[ident] = ts # add to cache
                if len(cache) > cache_size:
                    # Eviction Process
                    eviction_process(cache, int(cache_size/200), int(cache_size/4000), ts)



        hit_ratios.append(n_hits / length)    

    t3 = time.time()
    print('Time to Evaluate: ', str(t3-t2))
    #print(hit_ratios[:5])
    #print(hit_ratios[-5:])

    '''
    Plotting
    '''

    ml_x = cache_sizes

    # Inefficient but oh well
    opt_x = list(opt_hr_dict.keys())
    opt_x.sort()
    opt_hrs = list(opt_hr_dict.values())
    opt_hrs.sort()
    lru_x = list(lru_hr_dict.keys())
    lru_x.sort()
    lru_hrs = list(lru_hr_dict.values())
    lru_hrs.sort()
    rand_x = list(rand_hr_dict.keys())
    rand_x.sort()
    rand_hrs = list(rand_hr_dict.values())
    rand_hrs.sort()

    plt.figure(0)

    opt_curve, = plt.plot(opt_x, opt_hrs, 'r-')
    lru_curve, = plt.plot(lru_x, lru_hrs, 'g-')
    rand_curve, = plt.plot(rand_x, rand_hrs, 'k-')
    ml_curve, = plt.plot(ml_x, hit_ratios, 'b-')


    plt.xlabel('Cache Size')
    plt.ylabel('Hit Ratio')
    plt.legend((opt_curve, lru_curve, rand_curve, ml_curve), ('Optimal', 'LRU',
        "Random", '\"SmarterCache\"'), loc='lower right', markerscale=1.0)
    plt.savefig('eval/hrc/' + file_name + '_' + str(time.time()) + '.png')
    plt.close()
    

