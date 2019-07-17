import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random, math, time, sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from collections import deque
from PyMimircache.cacheReader.csvReader import CsvReader
from PyMimircache.cache.optimal import Optimal

'''
Initial Stuff
'''

file_name = sys.argv[1]
if sys.argv[2] == 'med':
    num = '1'
if sys.argv[2] == 'q3':
    num = '2'
q3_fut_rd_arr = pd.read_csv('fut_rd_stats/' + file_name + '_5000.csv')[num].to_numpy()
med_q3 = np.median(q3_fut_rd_arr)

damp_factor = med_q3

# Pretend to have perfect information
def avg_fut_rd(vtime, med=True):
    if not med:
        return q3_fut_rd_arr[int(vtime / 50)]
    else:
        return med_q3
    


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
        if self.training:
            output = torch.sigmoid((inputs - avg_rds)/damp_factor)
        else:
            output = inputs

        return output


'''
Data Processing Section
'''
# TODO: Fix this whole thing to get rid of t_now and stuff...
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
            next_access_dist.append(len(id_ser))
        next_access_time[req] = index
    
    next_access_dist.reverse()
    return next_access_dist


def create_dist_df(feature_df, samples, dists, start_time, eval=False):

    # Returns logistic virtual distance (sigmoid)
    def get_logit_dist(next_access_dist, timestamp):

        if next_access_dist != -1:
            return 1/(1 + np.exp(-(next_access_dist - avg_fut_rd(timestamp))/damp_factor))
        else: # return 1
            return 1

    
    train_data = []
    for t in samples:
        if not eval:
            logit_dist_val = get_logit_dist(dists[t], t)
        else:
            logit_dist_val = dists[t] + t # Actually the final ts for next req of that id
        
        delta = feature_df.shape[0] - t

        ser = feature_df.loc[t].to_list()
        for i in range(3,8):
            if ser[i] == 0:
                ser[i] = feature_df.shape[0]
        ser += [delta] + [logit_dist_val]
        train_data.append(ser)

    full_df = pd.DataFrame(data=train_data,
        columns=(list(range(N_FEATURES)) + ['final']))

    return full_df

def gen_train_eval_data(df):

    df = df.iloc[int(0.05*df.shape[0]):int(0.95*df.shape[0])].reset_index(drop=True)
    df_len = df.shape[0]
    df['vtime'] = df.index

    train_factor = random.uniform(0.6, 0.7)
    
    tc = time.time()
    n_samples = 500000

    time_samples = np.random.randint(0, int(train_factor * df_len), size=n_samples)
    learn_data = df.iloc[:int(train_factor * df_len)]
    train_dists = get_next_access_dist(learn_data['id'])
    
    train_df = create_dist_df(learn_data, time_samples, train_dists, 0)

    td = time.time()
    print('Time to Construct Training DataFrame:')
    print(td-tc)

    reader_params = {
        'label': 2,
        'real_time': 1
    }

    reader = CsvReader('ranktest/features/' + file_name + '_feat16.csv',
        init_params=reader_params)
    
    cache_size = 5000
    opt = Optimal(cache_size, reader)

    last_req_dict = {}

    #eval_data = df.iloc[int(train_factor * df_len):]
    #eval_dists = get_next_access_dist(eval_data['id'])
    dists = get_next_access_dist(df['id'])
    eval_dfs = []

    n_samples = 150
    time_stops = np.random.randint(int(train_factor*df_len), int(df_len*.95), size=n_samples)
    time_stops.sort()
    ret_time_stops = time_stops.copy()
    time_stops = deque(time_stops)

    for index, request in enumerate(df['id']):
        opt.access(request)
        last_req_dict[request] = index

        if time_stops and index == time_stops[0]:
            time_samples = []
            for req_id in opt.pq.keys():
                time_samples.append(last_req_dict[req_id])
            eval_df = create_dist_df(df, time_samples, dists, int(train_factor * df_len), eval=True)
            eval_dfs.append(eval_df)
            time_stops.popleft()
        
        if not time_stops:
            break

    te = time.time()
    print('Time to Construct Evaluation DataFrames:')
    print(te - td)

    return train_df, eval_dfs, ret_time_stops

'''
Pytorch Integration Section
'''
t1 = time.time()

train_df, eval_dfs, times = gen_train_eval_data(pd.read_csv('ranktest/features/' + file_name + '_feat16.csv'))
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

eval_feats = []
eval_targets = []

for eval_df in eval_dfs:
    eval_feat = eval_df.drop(columns=[0,1,
        'final']).astype('float64').to_numpy()
    eval_target = eval_df['final'].astype('float64').to_numpy()

    eval_feat = np.concatenate((eval_feat[:,[0]], normalizing_func(eval_feat[:,1:])), axis=1)

    eval_feat = torch.tensor(eval_feat, dtype=torch.float)
    
    eval_feats.append(eval_feat)
    eval_targets.append(eval_target)
    #eval_target = torch.tensor(eval_target, dtype=torch.float)

model = CacheNet(p=0.5)
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.2)

lambda1 = lambda epoch: 0.99
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
#train_feat = torch.randn(len(train_target), 34)

model.train()
print(train_target)
for t in range(300):
    # Forward Pass
    y_pred = model(train_feat)

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
    # Training Score
    y_pred = model(train_feat)
    print('Training Score:')
    print(criterion(y_pred, train_target))

    model.eval()
    num_evicted = 100
    model_predictions = []
    rec_predictions = []

    for i in range(len(eval_feats)):
        y_pred = model(eval_feats[i])
        pred_lst = y_pred.numpy().flatten()
        pred_lst += eval_feats[i][:,0].numpy() # add the vtimes in

        # Gives indices for the num_evicted items that will be evicted by model
        pred_evict_inds = np.argsort(pred_lst)[-1*num_evicted:]

        # Gives indices for the num_evicted items that will be evicted by pure recency
        rec_evict_inds = np.argsort(eval_feats[i][:,0].numpy())[:num_evicted]

        # Actual times for model evicted items
        actual_times_model = [eval_targets[i][index] for index in pred_evict_inds]

        # Actual times for recency evicted items
        actual_times_rec = [eval_targets[i][index] for index in rec_evict_inds]
        
        # Create a dict for actual time -> index in sorted list (lower is better)
        sorted_times = eval_targets[i].tolist()
        sorted_times.sort(reverse=True)
        time_dict = {}
        for index, tm in enumerate(sorted_times):
            time_dict[tm] = index

        # Get the ranks for the evicted items by model
        curr_pred_model = [time_dict[tm] for tm in actual_times_model]

        # Get the ranks for the evicted items by recency
        curr_pred_rec = [time_dict[tm] for tm in actual_times_rec]
        
        model_predictions.append(np.mean(curr_pred_model))
        rec_predictions.append(np.mean(curr_pred_rec))
        #print(predictions)

    t2 = time.time()
    print('Time to Train:')
    print(str(t2-t1))

    plt.figure(0)
    plt.scatter(times, model_predictions, s=4.0, c='g', edgecolors='none')
    plt.scatter(times, rec_predictions, s=4.0, c='b', edgecolors='none')
    plt.savefig('eval/' + file_name + '_' + sys.argv[2] + '_' + str(time.time()) + '.png')
    plt.close()


