import pandas as pd 
import numpy as np
import random, math, time, bisect, sys
import scipy.stats
from collections import defaultdict
from sklearn.metrics import make_scorer
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


def scorer_func(X, Y):
    length = len(Y)
    total = 0
    for i in range(length):
        if (X[i] == 0 and Y[i] == 0) or (X[i] == 1 and Y[i] == 1):
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

    # Returns 1 if a comes before b, 0 if b comes before a, -1 otherwise
    def weird_ladder_func(id_a_lst, id_b_lst, t_now, t_max):
        
        a_cand_ind = bisect.bisect(id_a_lst, t_now)
        b_cand_ind = bisect.bisect(id_b_lst, t_now)
        # Indicators for having a hit between t_now and t_max
        a_bool = not (a_cand_ind == len(id_a_lst) or id_a_lst[a_cand_ind] > t_max)
        b_bool = not (b_cand_ind == len(id_b_lst) or id_b_lst[b_cand_ind] > t_max)
        if a_bool and b_bool:
            if id_a_lst[a_cand_ind] < id_b_lst[b_cand_ind]:
                return 1
            else:
                return 0
        elif a_bool: # only a
            return 1
        elif b_bool: # only b
            return 0
        else: # neither
            return -1


    train_data = []
    for t_a, t_b in pair_set:
        id_a = feature_df.at[t_a, 'id']
        id_b = feature_df.at[t_b, 'id']
        comp_val = weird_ladder_func(req_dict[id_a], req_dict[id_b], t_now, t_max)
        if comp_val != -1:
            delta_a = math.sqrt(t_now - t_a)
            delta_b = math.sqrt(t_now - t_b)

            ser = feature_df.loc[t_a].to_list() + [delta_a] + feature_df.loc[t_b].to_list() + [delta_b, comp_val]
            train_data.append(ser)

    full_df = pd.DataFrame(data=train_data,
        columns=(list(range(N_FEATURES * 2)) + ['final']))

    return full_df

def gen_train_eval_data(df):
    
    train_df = pd.DataFrame(columns=(list(range(N_FEATURES * 2)) + ['final']))
    req_dict = defaultdict(list)

    df = df.iloc[int(0.05*df.shape[0]):int(0.95*df.shape[0])].reset_index(drop=True)
    df_len = df.shape[0]

    for index, id in enumerate(df['id'].to_numpy()):
        # For later stuff
        req_dict[id].append(index)

    n_time_samples = 40
    train_factor = random.uniform(0.75, 0.8)
    
    times = set()
    while (len(times) < n_time_samples): 
        eval_len = int((1-train_factor) * random.uniform(0.9, 1.1) * df_len)
        rand = random.randint(0, int(train_factor*df_len)-eval_len)
        times.add((rand, rand+eval_len))
    
    tc = time.time()

    n_pairs = 5000
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
    n_eval_pairs = 100000
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

    lru_predictions = np.empty([eval_df.shape[0],1], dtype=int)
    LRU_ind_dict = {}
    for index, row in eval_df[[1, (1 + N_FEATURES)]].iterrows():
        id_a = row[1]
        id_b = row[1 + N_FEATURES]

        if id_a not in LRU_ind_dict:
            id_a_cand_ind = bisect.bisect_left(req_dict[id_a], t_now) - 1
            LRU_ind_dict[id_a] = req_dict[id_a][id_a_cand_ind]
        if id_b not in LRU_ind_dict:
            id_b_cand_ind = bisect.bisect_left(req_dict[id_b], t_now) - 1
            LRU_ind_dict[id_b] = req_dict[id_b][id_b_cand_ind]
        
        # A is least recently used (favor b, kick a, predict b comes before a in future)
        if LRU_ind_dict[id_a] < LRU_ind_dict[id_b]:
            lru_predictions[index, 0] = 0
        else: # B is least recently used, (favor a, kick b, predict a comes before b in future)
            lru_predictions[index, 0] = 1
    
    tg = time.time()
    print('Time to Construct LRU Stuff:')
    print(tg - tf)

    return train_df, eval_df, lru_predictions


def learn(classifier, train_df):
    train_target = train_df['final'].astype('int').to_numpy()
    train_features = train_df.drop(columns=[0,1,2,3,4,5,(0 + N_FEATURES),
        (1 + N_FEATURES),(2 + N_FEATURES),(3 + N_FEATURES),
        (4 + N_FEATURES),(5 + N_FEATURES),'final']).to_numpy()

    t4 = time.time()

    classifier.fit(train_features, train_target)
    print('Gradient Boosting Tree Fit Score:')
    learn5 = np.array(classifier.predict(train_features))
    print(scorer_func(train_target, learn5))

    t5 = time.time()
    print('Time to train:')
    print(t5-t4)

    print('Feature Importances:')
    print(classifier.feature_importances_)

    return classifier


def evaluate(classifier, eval_df, lru_pred, write=False):

    eval_target = eval_df['final'].astype('int').to_numpy()
    eval_features = eval_df.drop(columns=[0,1,2,3,4,5,(0 + N_FEATURES),
        (1 + N_FEATURES),(2 + N_FEATURES),(3 + N_FEATURES),
        (4 + N_FEATURES),(5 + N_FEATURES),'final']).to_numpy()

    print('Light GBM Score:')
    pred5 = np.array(classifier.predict(eval_features))
    print(scorer_func(eval_target, pred5))

    if write:
        eval_df['predicted'] = classifier.predict(eval_features)
        eval_df.iloc[:1000].to_csv('features/results/temp_res.csv', index=False)

    # Comparison to LRU
    print('Recency Evaluation Score:')
    print(scorer_func(lru_pred, eval_target))   

def tune(df):

    param_test = {
        'reg_alpha': [1e-2, .1, 1, 10],
        'reg_lambda': [1e-2, .1, 1, 10]
    }

    scorer = make_scorer(scorer_func)

    classifier_test = LGBMClassifier(boosting_type='dart', n_estimators=100,
        objective='binary', learning_rate=0.07,
        subsample=0.8, max_depth=150, num_leaves=400, n_jobs=1)
    cv = GridSearchCV(estimator=classifier_test, param_grid=param_test, scoring=scorer,
        verbose=3, cv=3
    )
    
    target = df['final'].astype('float').to_numpy()
    features = df.drop(columns=[0,1,2,(0 + N_FEATURES),
        (1 + N_FEATURES),(2 + N_FEATURES),'final']).to_numpy()

    cv.fit(features, target)
    print(cv.best_score_)
    print(cv.best_params_)
    f = open('cv_results.txt', 'w+')
    f.write(str(cv.best_score_))
    f.write(str(cv.best_params_))
    f.write(str(cv.cv_results_))
    f.close()

file_name = sys.argv[1]
train_df, eval_df, lru_predictions = gen_train_eval_data(pd.read_csv('features/' + file_name + '.csv'))

classifier = LGBMClassifier(boosting_type='dart', learning_rate=0.07, n_estimators=200,
    objective='binary',
    subsample=0.8, max_depth=150, num_leaves=500, n_jobs=1, reg_alpha=10, min_child_samples=30)

classifier = learn(classifier, train_df)

evaluate(classifier, eval_df, lru_predictions, write=True)

