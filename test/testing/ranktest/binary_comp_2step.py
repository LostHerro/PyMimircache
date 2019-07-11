import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random, math, bisect, time, sys
from collections import defaultdict
from sklearn.preprocessing import normalize

# To be implemented
def avg_fut_rd(vtime):
    return random.random()

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

    def forward(self, inputs):
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
        output = torch.atan(self.out_layer(inputs))

        return output


def scorer_func(X, Y):
    length = len(Y)
    total = 0
    for i in range(length):
        if (X[i,0] < 0.5 and Y[i,0].item() == 0) or (X[i,0].item() >= 0.5 and Y[i,0].item() == 1):
            total += 1
    return total / length


'''
Data Processing Section
'''

N_FEATURES = 23

def random_pair_generator(n, low, high, min_separation=0, sorted_=False):
    pair_set = set()
    while len(pair_set) < n:
        pair = random.sample(range(low, high), 2)
        if sorted_:
            pair = sorted(pair)
        pair = tuple(pair)
        if min_separation > 0:
            if abs(pair[0] - pair[1]) > min_separation:
                pair_set.add(pair)
        else:
            pair_set.add(pair)
    return pair_set

def create_dist_df(feature_df, samples, t_now, t_max, req_dict):

    # Returns logistic virtual distance (arctan) after t_now
    def get_logit_dist(id_lst, t_now, t_max):
        
        cand_ind = bisect.bisect(id_lst, t_now)
        # Indicators for having a hit between t_now and t_max

        if not (cand_ind == len(id_lst) or id_lst[cand_ind] > t_max):
            return np.arctan(id_lst[cand_ind] - avg_fut_rd(t_now))
        else: # return pi/2
            return np.pi / 2


    train_data = []
    for t in samples:
        id_t = feature_df.at[t, 'id']
        logit_dist_val = get_logit_dist(req_dict[id_t], t_now, t_max)
        delta_a = math.sqrt(t_now - t)

        ser = feature_df.loc[t].to_list() + [delta_a] + [logit_dist_val]
        train_data.append(ser)

    full_df = pd.DataFrame(data=train_data,
        columns=(list(range(N_FEATURES)) + ['final']))

    return full_df

def gen_train_eval_data(df):
    
    train_df = pd.DataFrame(columns=(list(range(N_FEATURES * 2)) + ['final']))
    req_dict = defaultdict(list)

    df = df.iloc[int(0.05*df.shape[0]):int(0.95*df.shape[0])].reset_index(drop=True)
    df_len = df.shape[0]

    for index, id in enumerate(df['id'].to_numpy()):
        # For later stuff
        req_dict[id].append(index)

    n_windows = 40
    train_factor = random.uniform(0.75, 0.8)
    
    times = set()
    while (len(times) < n_windows): 
        eval_len = int((1-train_factor) * random.uniform(0.9, 1.1) * df_len)
        rand = random.randint(0, int(train_factor*df_len)-eval_len)
        times.add((rand, rand+eval_len))
    
    tc = time.time()

    n_samples_window = 5000
    for tm in times:
        alpha = random.uniform(0.4, 0.6)
        t_now = int(tm[0]*alpha + tm[1]*(1-alpha))

        time_samples = np.random.randint(tm[0], t_now, size=n_samples_window)

        curr_df = create_dist_df(df.iloc[tm[0]:t_now],
            time_samples, t_now, tm[1], req_dict)
        
        train_df = pd.concat([train_df, curr_df], sort=False)

    td = time.time()
    print('Time to Construct Training DataFrame:')
    print(td-tc)

    te = time.time()
    n_eval_reqs = 100000
    alpha = random.uniform(0.4, 0.6)
    t_now = int(alpha*train_factor*df_len + (1-alpha)*df_len)
    time_samples = np.random.randint(int(train_factor * df_len), t_now, size=n_eval_reqs)
    eval_data = df.iloc[int(train_factor * df_len):t_now]

    eval_df = create_dist_df(eval_data, time_samples,
        t_now, df_len, req_dict)
    tf = time.time()
    print('Time to Construct Evaluation DataFrame:')
    print(tf - te)

    return train_df, eval_df

'''
Pytorch Integration Section
'''

file_name = sys.argv[1]
train_df, eval_df = gen_train_eval_data(pd.read_csv('features/' + file_name + '.csv'))
normalizing_func = lambda x: (x-np.mean(x, axis=0))/np.std(x, axis=0)

train_feat = train_df.drop(columns=[0,1,2,3,4,5,(0 + N_FEATURES),
    (1 + N_FEATURES),(2 + N_FEATURES),(3 + N_FEATURES),(4 + N_FEATURES),(5 + N_FEATURES),
    'final']).astype('float64').to_numpy()
train_target = train_df[['final']].astype('float64').to_numpy()

train_feat = normalizing_func(train_feat)

train_feat = torch.tensor(train_feat, dtype=torch.float)
train_target = torch.tensor(train_target, dtype=torch.float)

eval_feat = eval_df.drop(columns=[0,1,2,3,4,5,(0 + N_FEATURES),
    (1 + N_FEATURES),(2 + N_FEATURES),(3 + N_FEATURES),(4 + N_FEATURES),(5 + N_FEATURES),
    'final']).astype('float64').to_numpy()
eval_target = eval_df[['final']].astype('float64').to_numpy()

eval_feat = normalizing_func(eval_feat)

eval_feat = torch.tensor(eval_feat, dtype=torch.float)
eval_target = torch.tensor(eval_target, dtype=torch.float)

model = CacheNet(p=0.5)
criterion = torch.nn.MSELoss
optimizer = optim.Adam(model.parameters(), lr=0.02)

lambda1 = lambda epoch: 0.99
#scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
#train_feat = torch.randn(len(train_target), 34)

print(train_target)
for t in range(200):
    # Forward Pass
    y_pred = model(train_feat)

    # Loss
    loss = criterion(y_pred, train_target)
    if (t % 5 == 0):
        print(t, loss.item())
    if (t % 10 == 0):
        print(y_pred)

    # Backward Pass And Update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #scheduler.step()

# Evaluation of Model
model.eval()
with torch.no_grad():
    # Training Score
    y_pred = model(train_feat)
    print('Training Score:')
    print(scorer_func(y_pred, train_target))

    # Evaluation Score
    y_pred = model(eval_feat)
    print('Evaluation Score:')
    print(scorer_func(y_pred, eval_target))

print(y_pred)


'''
Binary Classification Evaluation
'''
