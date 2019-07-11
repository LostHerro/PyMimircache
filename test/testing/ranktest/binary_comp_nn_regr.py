import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random, math, bisect, time
from collections import defaultdict
from sklearn.preprocessing import normalize

class CacheNet(nn.Module):

    N_TRUE_FEATURES = 34

    def __init__(self, p=0.0):
        super(CacheNet, self).__init__()
        self.in_layer = nn.Linear(self.N_TRUE_FEATURES, 20)
        self.in_drop = nn.Dropout(p=p)
        self.h1_layer = nn.Linear(20, 10)
        self.h1_drop = nn.Dropout(p=p)
        self.h2_layer = nn.Linear(10, 4)
        self.h2_drop = nn.Dropout(p=p)
        self.out_layer = nn.Linear(4, 1)

    def forward(self, inputs):
        inputs = F.relu(self.in_layer(inputs))
        inputs = self.in_drop(inputs)
        inputs = F.relu(self.h1_layer(inputs))
        inputs = self.h1_drop(inputs)
        inputs = F.relu(self.h2_layer(inputs))
        inputs = self.h2_drop(inputs)
        output = self.out_layer(inputs)

        return output



# Current function: L(x1,x2) = c(x1-x2)^2 - x1x2/sqrt(x1^2 + x2^2)
# x1 is y, x2 is y_pred
# Should be fine since x1 is never zero
class Loss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, y_pred, y):
        c = 1.0
        ctx.save_for_backward(y_pred, y)
        loss = (c * (y - y_pred)**(2) - y * y_pred / (y**(2) + y_pred**(2))**(0.5)).sum()/(len(y))
        return loss

    # ???
    @staticmethod
    def backward(ctx, grad_output):
        c = 1.0
        y_pred, y = ctx.saved_tensors
        grad_input = (-y**(3)/((y**(2) + y_pred**(2))**(1.5)) - 2*c*(y - y_pred))/len(y)
        return grad_input, None


""" class Loss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, y_pred, y):
        ctx.save_for_backward(y_pred, y)
        loss = ((y - y_pred)**(2)).sum()/len(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y_pred, y = ctx.saved_tensors
        grad_input = -2*(y - y_pred)/len(y)
        return grad_input, None """


""" def loss_func(y_pred, y):
    c = 0.3
    loss = (c * (y - y_pred)**(2) - y * y_pred / (y**(2) + y_pred**(2))**(0.5)).sum()
    return loss """

def scorer_func(X, Y):
    length = len(X)
    total = 0
    for i in range(length):
        if (X[i,0].item() > 0 and Y[i,0].item() > 0) or (X[i,0].item() < 0 and Y[i,0].item() < 0):
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
    
def create_comp_df(feature_df, pair_set, t_now, t_max, req_dict):

    # Returns a - b virtual distance for first hit between t_now and t_max
    def weird_ladder_func(id_a_lst, id_b_lst, t_now, t_max):
        
        a_cand_ind = bisect.bisect(id_a_lst, t_now)
        b_cand_ind = bisect.bisect(id_b_lst, t_now)
        # Indicators for having a hit between t_now and t_max
        a_bool = not (a_cand_ind == len(id_a_lst) or id_a_lst[a_cand_ind] > t_max)
        b_bool = not (b_cand_ind == len(id_b_lst) or id_b_lst[b_cand_ind] > t_max)
        if a_bool and b_bool:
            return id_a_lst[a_cand_ind] - id_b_lst[b_cand_ind]
        elif a_bool: # only a
            return id_a_lst[a_cand_ind] - t_max
        elif b_bool: # only b
            return t_max - id_b_lst[b_cand_ind]
        else: # neither
            return 0

    train_data = []
    for t_a, t_b in pair_set:
        id_a = feature_df.at[t_a, 'id']
        id_b = feature_df.at[t_b, 'id']
        comp_val = weird_ladder_func(req_dict[id_a], req_dict[id_b], t_now, t_max)
        if comp_val != 0:
            delta_a = math.sqrt(t_now - t_a)
            delta_b = math.sqrt(t_now - t_b)
            if comp_val > 0:
                comp_val = np.log1p(comp_val)
            else:
                comp_val = -np.log1p(-comp_val)
            ser = feature_df.loc[t_a].to_list() + [delta_a] + feature_df.loc[t_b].to_list() + [delta_b, comp_val]
            train_data.append(ser)

    full_df = pd.DataFrame(data=train_data,
        columns=(list(range(N_FEATURES * 2)) + ['final']))

    return full_df

def gen_train_eval_data(df):
    
    train_df = pd.DataFrame(columns=(list(range(N_FEATURES * 2)) + ['final']))
    req_dict = defaultdict(list)

    # Discard the first and last 5% of data due to boundary issues
    df = df.iloc[int(0.05*df.shape[0]):int(0.95*df.shape[0])].reset_index(drop=True)
    df_len = df.shape[0]

    for index, id in enumerate(df['id'].to_numpy()):
        # For later stuff
        req_dict[id].append(index)

    n_time_samples = 40
    train_factor = random.uniform(0.75, 0.8)
    #sep_factor = 0.1
    
    times = set()
    while (len(times) < n_time_samples): 
        eval_len = int((1-train_factor) * random.uniform(0.9, 1.1) * df_len)
        rand = random.randint(0, int(train_factor*df_len)-eval_len)
        times.add((rand, rand+eval_len))
    #times = __random_pair_generator(n_time_samples, 0, int(train_factor * df_len), 
    #    min_separation=int(sep_factor * df_len), sorted_=True)
    
    tc = time.time()

    n_pairs = 10000
    for tm in times:
        alpha = random.uniform(0.4, 0.6)
        t_now = int(tm[0]*alpha + tm[1]*(1-alpha))
        comp_pairs = random_pair_generator(n_pairs, tm[0], t_now, sorted_=False)
        curr_df = create_comp_df(df.iloc[tm[0]:t_now],
            comp_pairs, t_now, tm[1], req_dict)
        
        train_df = pd.concat([train_df, curr_df], sort=False)
    td = time.time()
    print('Time to Construct Training DataFrame:')
    print(td-tc)

    te = time.time()
    n_eval_pairs = 150000
    alpha = random.uniform(0.4, 0.6)
    t_now = int(alpha*train_factor*df_len + (1-alpha)*df_len)
    comp_eval_pairs = random_pair_generator(n_eval_pairs, 
        int(train_factor * df_len), t_now, sorted_=False)
    eval_data = df.iloc[int(train_factor * df_len):t_now]
    eval_df = create_comp_df(eval_data, comp_eval_pairs,
        t_now, df_len, req_dict)
    tf = time.time()
    print('Time to Construct Evaluation DataFrame:')
    print(tf - te)

    return train_df, eval_df

'''
Pytorch Integration Section
'''

train_df, eval_df = gen_train_eval_data(pd.read_csv('traces/lax_1448_small_feat23_100k.csv'))
normalizing_func = lambda x: (x-np.mean(x, axis=0))/np.std(x, axis=0)

train_feat = train_df.drop(columns=[0,1,2,3,4,5, (0 + N_FEATURES),
    (1 + N_FEATURES),(2 + N_FEATURES),(3 + N_FEATURES), (4 + N_FEATURES), (5 + N_FEATURES),
    'final']).astype('float64').to_numpy()
train_target = train_df[['final']].astype('float64').to_numpy()

train_feat = normalizing_func(train_feat)

train_feat = torch.tensor(train_feat, dtype=torch.float)
train_target = torch.tensor(train_target, dtype=torch.float)

eval_feat = eval_df.drop(columns=[0,1,2,3,4,5, (0 + N_FEATURES),
    (1 + N_FEATURES),(2 + N_FEATURES),(3 + N_FEATURES), (4 + N_FEATURES), (5 + N_FEATURES),
    'final']).astype('float64').to_numpy()
eval_target = eval_df[['final']].astype('float64').to_numpy()

eval_feat = normalizing_func(eval_feat)

eval_feat = torch.tensor(eval_feat, dtype=torch.float)
eval_target = torch.tensor(eval_target, dtype=torch.float)

model = CacheNet(p=0.5)
criterion = torch.nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
#optimizer = optim.Adam(model.parameters(), lr=0.01)

lambda1 = lambda epoch: 0.98
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda1)

print(train_target)
for t in range(500):
    # Forward Pass
    y_pred = model(train_feat)

    # Loss
    #loss_func = Loss.apply 
    #loss = loss_func(y_pred, train_target)
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
