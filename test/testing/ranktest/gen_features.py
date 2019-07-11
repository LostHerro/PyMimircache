import numpy as np
import pandas as pd
import time, random
from collections import defaultdict, deque

def generate_features(df, sample=0, hexadecimal=False):

    if sample != 0:
        if hexadecimal:
            char_lst = [str(i) for i in range(10)] + ['a', 'b', 'c', 'd', 'e']
        else:
            char_lst = [str(i) for i in range(10)]
        take = random.sample(char_lst, sample)
        substring_ser = df['id'].apply(lambda x: x[-1:])
        df = df[substring_ser.isin(take)]
        df = df.reset_index(drop=True)
    
    train_len = df.shape[0]

    # Reuse Distance Statistics Features
    last_req_dict = {}
    dist_dict = defaultdict(list)
    stat_arr = np.zeros((train_len, 8), dtype=np.float64)
    
    # Frequency Features
    freq_arr = np.ones((train_len, 5), dtype=np.float64)
    freq2_size = 4
    freq4_size = 16
    freq6_size = 64
    freq8_size = 256
    freq10_size = 1024
    freq_size_lst = [freq2_size, freq4_size, freq6_size, freq8_size, freq10_size]

    freq2_flg = False
    freq4_flg = False
    freq6_flg = False
    freq8_flg = False
    freq10_flg = False
    freq_flg_lst = [freq2_flg, freq4_flg, freq6_flg, freq8_flg, freq10_flg]
    
    freq2_deq = deque(maxlen=freq2_size)
    freq4_deq = deque(maxlen=freq4_size)
    freq6_deq = deque(maxlen=freq6_size)
    freq8_deq = deque(maxlen=freq8_size)
    freq10_deq = deque(maxlen=freq10_size)
    freq_deq_lst = [freq2_deq, freq4_deq, freq6_deq, freq8_deq, freq10_deq]

    freq2_dict = {}
    freq4_dict = {}
    freq6_dict = {}
    freq8_dict = {}
    freq10_dict = {}
    freq_dict_lst = [freq2_dict, freq4_dict, freq6_dict, freq8_dict, freq10_dict]

    # Request Rate Features
    req_rate_arr = np.zeros((train_len, 3), dtype=np.float64)
    req2_flg = False
    req4_flg = False
    req6_flg = False
    req_flg_lst = [req2_flg, req4_flg, req6_flg]


    def next_req_deletion(freq_deq, freq_dict):
        id_to_remove = freq_deq.popleft()
        deq = freq_dict[id_to_remove]
        if len(deq) == 1:
            del freq_dict[id_to_remove]
        else:
            deq.popleft()

    def next_req_insertion(dict_ind, freq_deq, freq_dict, req_id, req_vtime, freq_arr):
        freq_deq.append(req_id)

        if req_id not in freq_dict:
            freq_dict[req_id] = deque([req_vtime])
        else:
            freq_dict[req_id].append(req_vtime)

        for vtime in freq_dict[req_id]:
            freq_arr[vtime, dict_ind] += 1


    for index, series in df.iterrows():
        id = series['id']

        # Populate Reuse Distance Dict
        if id in last_req_dict:
            dist_dict[id].append(index - last_req_dict[id])
        last_req_dict[id] = index

        # Populate Frequency Array
        for i in range(5):
            if index == freq_size_lst[i]:
                freq_flg_lst[i] = True
            
            if freq_flg_lst[i]:
                next_req_deletion(freq_deq_lst[i], freq_dict_lst[i])
            next_req_insertion(i, freq_deq_lst[i], freq_dict_lst[i],
                id, index, freq_arr)

        # Populate Request Rate Array
        for i in range(3):
            if index == freq_size_lst[i]/2:
                req_flg_lst[i] = True
            elif index == train_len - freq_size_lst[i]/2:
                req_flg_lst[i] = False
            
            if req_flg_lst[i]:
                time_delta = max(0.001, df.loc[index + freq_size_lst[i]/2]['time']
                    - df.loc[index - freq_size_lst[i]/2]['time'])
                req_rate_arr[index, i] = freq_size_lst[i] / time_delta

    # Convert to float frequencies
    for i in range(5):
        freq_arr[:,i] /= freq_size_lst[i]


    def __generate_stats(distances):
        min_ = np.min(distances)
        max_ = np.max(distances)
        mean = np.mean(distances)
        median = np.median(distances)
        range_ = np.ptp(distances)
        std = np.std(distances)
        q1 = np.percentile(distances, 25)
        q3 = np.percentile(distances, 75)

        return (min_, max_, mean, median, range_, std, q1, q3)
    
    for index, id in df['id'].iteritems():
        min_, max_, mean, median, range_, std, q1, q3 = tuple([train_len] * 8)
        if dist_dict.get(id, []):
            min_, max_, mean, median, range_, std, q1, q3 = __generate_stats(dist_dict[id])
        
        stat_arr[index] = [min_, max_, mean, median, range_, std, q1, q3]

    df['vtime'] = df.index.to_numpy() + 1

    df['access_day'] = df['time'].apply(
        lambda x: int(x % 6.048e5 / 8.64e4))

    df['access_hr'] = df['time'].apply(
        lambda x: int(x % 8.64e4 / 3600))

    df['access_min'] = df['time'].apply(
        lambda x: int(x % 3600 / 60))

    f = lambda x: np.arctan(x/500)
    stat_arr = f(stat_arr)

    df = df.join(pd.DataFrame(stat_arr, 
        columns=['dist_min', 'dist_max', 'dist_mean', 'dist_median',
        'dist_range', 'dist_std', 'dist_q1', 'dist_q3']))

    df = df.join(pd.DataFrame(freq_arr,
        columns=['freq4', 'freq16', 'freq64', 'freq256', 'freq1024']))
    
    df = df.join(pd.DataFrame(req_rate_arr,
        columns=['req_rate4', 'req_rate16', 'req_rate64']))

    return df

def main():
    
    source_file = 'traces/lax_1448_small'
    target_dest = 'features/lax_1448_small_test.csv'

    col_names = ['time', 'id']
    data_types = {'time': 'float64', 'id': 'str'}

    df = pd.read_csv(source_file, sep='\t', usecols=[0,5], header=0, names=col_names, dtype=data_types
        ,nrows=10000
        )
    
    ta = time.time()
    print('Generating Features for ' + source_file + ':')

    df = generate_features(df, sample=0, hexadecimal=False)
    df.to_csv(target_dest, index=False)

    tb = time.time()
    print('Done. Time Elapsed:')
    print(tb - ta)


if __name__ == '__main__':
    main()