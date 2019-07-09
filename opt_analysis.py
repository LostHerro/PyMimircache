import numpy as np
import pandas as pd
import random, sys, time
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from PyMimircache.cacheReader.csvReader import CsvReader
from PyMimircache.cache.optimal import Optimal

def main():
    init_params = {'label': 2, 'real_time': 1}
    file_name = sys.argv[1]
    
    # Write modes: 'rd' for future reuse distance plot,
    #              'auto' for autocorrelation plot,
    #              'ev' for eviction distance plot,
    #              'rd_csv' for future reuse distance csv,
    #              'all' for everything
    write_modes = set(sys.argv[2:])
    if 'all' in write_modes:
        write_modes = {'rd', 'auto', 'ev', 'rd_csv'}

    reader = CsvReader('/research/george/ranktest/traces/shared/' + file_name + '_no_feat.csv',
        init_params=init_params)

    cache_size = 5000
    rand_factor = 0.005 # Take X% of values in the cache to analyze
    opt = Optimal(cache_size, reader)
    
    # Timestamp values
    fut_rd_arr1 = []
    # Future Reuse values
    fut_rd_arr2 = []

    # Median values
    med_x = []
    med_y = []
    q3_y = []

    # Evict Distance values
    ev_x = []
    ev_y = []

    #Dict to store the last vtime for a given id
    id_to_vtime = {}

    t0 = time.time()
    for request in reader:
        ev_item = []
        opt.access(request, evict_item_list=ev_item)
        id_to_vtime[request] = opt.ts
        temp_lst = []
        if 'rd' in write_modes or 'rd_csv' in write_modes:
            for req_id in opt.pq:
                if random.random() < rand_factor:
                    #fut_rd_arr1.append(opt.ts)
                    #fut_rd_arr2.append(-opt.pq[req_id] - opt.ts)
                    temp_lst.append(-opt.pq[req_id] - opt.ts) 
        if temp_lst:
            med_x.append(opt.ts)
            med_y.append(np.median(temp_lst))
            q3_y.append(np.percentile(temp_lst, 75))
        if 'ev' in write_modes:
            if ev_item and opt.ts != id_to_vtime[ev_item[0]]:
                ev_x.append(opt.ts)
                ev_y.append(opt.ts - id_to_vtime[ev_item[0]])
        if (opt.ts % 100000 == 0):
            print('progress', opt.ts)

    t1 = time.time()  
    print('Time for Iteration: ', str(t1-t0))

    """ x = fut_rd_arr1 # virtual time
    y = fut_rd_arr2 # future reuse distance

    # Scatter plot of virtual time and future reuse distance
    plt.figure(1)

    samples = np.random.randint(0, len(x), size=int(0.01*len(x))) # Samples 1% of values to plot
    x_sample = [x[i] for i in samples]
    y_sample = [y[i] for i in samples]

    plt.scatter(x_sample, y_sample, s=0.2)
    plt.title('Scatter Plot of Future Reuse Distances and Virtual Time')
    plt.xlabel('Virtual Time')
    plt.ylabel('Future Virtual Distance')
    plt.savefig('img/scatter_' + file_name + '_size_' + str(cache_size) + '.png')

    t2 = time.time()
    print('Time for Scatter Plot: ', str(t2-t1)) """

    # 2D Histogram
    """ plt.figure(2)

    plt.hist2d(x, y, bins=[opt.ts//100, 25], range=[[0, opt.ts], [0, cache_size*10]])
    plt.title('2D Histogram of Future Reuse Distance Frequency and Virtual Time')
    plt.xlabel('Virtual Time')
    plt.ylabel('Future Virtual Distance')
    plt.savefig('img/hist_' + file_name + '_size_' + str(cache_size) + '.png')

    t3 = time.time()
    print('Time for 2D Histogram: ', str(t3-t2)) """

    # Median Future Reuse Distance
    
    if 'rd' in write_modes or 'rd_csv' in write_modes:
        plt.figure(3)
        length = len(med_x)
        med_x = med_x[int(0.05*length):int(0.95*length)]
        med_y = med_y[int(0.05*length):int(0.95*length)]
        q3_y = q3_y[int(0.05*length):int(0.95*length)]

        bucket_size = 50
        temp_y_med_lst = []
        temp_y_q3_lst = []
        avg_x_fut_dist = []
        avg_y_med_fut_dist = []
        avg_y_q3_fut_dist = []
        for index, dist in enumerate(med_y):
            if index % bucket_size == bucket_size - 1:
                avg_x_fut_dist.append(med_x[index])
                avg_y_med_fut_dist.append(np.mean(temp_y_med_lst))
                #avg_y_q3_fut_dist.append(np.percentile(temp_y_med_lst, 75))
                avg_y_q3_fut_dist.append(np.mean(temp_y_q3_lst))
                temp_y_med_lst = []
                temp_y_q3_lst = []

            temp_y_med_lst.append(dist)
            temp_y_q3_lst.append(q3_y[index])

        """ samples = np.random.randint(0, len(med_x), size=int(0.05*len(med_x))) # Samples X% of values to plot
        med_x = [med_x[i] for i in samples]
        med_y = [med_y[i] for i in samples]
        q3_y = [q3_y[i] for i in samples] """

        if 'rd' in write_modes:
            fut_rd_med = plt.scatter(avg_x_fut_dist, avg_y_med_fut_dist, s=0.2, c='b', edgecolors='none')
            plt.title('Scatter Plot of Future Reuse Distance Stats and Virtual Time')
            plt.xlabel('Virtual Time')
            plt.ylabel('Future Virtual Distance')

            fut_rd_q3 = plt.scatter(avg_x_fut_dist, avg_y_q3_fut_dist, s=0.2, c='g', edgecolors='none')
            plt.legend((fut_rd_med, fut_rd_q3), ('Median', 'Q3'), loc='upper right', markerscale=5.0)

            plt.savefig('img/fut_rd_stats_' + file_name + '_size_' + str(cache_size) + '_' + str(time.time()) + '.png')

        if 'rd_csv' in write_modes:
            ser_x = pd.Series(data=avg_x_fut_dist)
            ser_med = pd.Series(data=avg_y_med_fut_dist)
            ser_q3 = pd.Series(data=avg_y_q3_fut_dist)
            df = pd.concat([ser_x, ser_med, ser_q3], axis=1)
            df.to_csv('img/fut_rd_stats/' + file_name + '_' + str(cache_size) + '.csv', index=False)
    
        plt.close()

    t4 = time.time()
    print('Time for Median/Q3 Plot: ', str(t4-t1))
    

    # Autocorrelation :(
    if 'auto' in write_modes:
        plt.figure(4)
        fig, axs = plt.subplots(nrows=2)

        #reduction_factor = 123
        #avg_y_med_fut_dist = [avg_y_med_fut_dist[i] for i in range(len(avg_y_med_fut_dist)) if i % reduction_factor == 0]
        #avg_y_q3_fut_dist = [avg_y_q3_fut_dist[i] for i in range(len(avg_y_q3_fut_dist)) if i % reduction_factor == 0]
        
        axs[0].acorr(avg_y_med_fut_dist, maxlags=20000, normed=True)
        axs[1].acorr(avg_y_q3_fut_dist, maxlags=20000, normed=True)
        for ax in axs:
            ax.set(ylabel='Future Virtual Distance')
        axs[1].set(xlabel='Lags (Virtual Time / 50)')
        axs[0].set_title('Future RD Median Auto-Correlation')
        axs[1].set_title('Future RD Q3 Auto-Correlation')
        fig.savefig('img/fut_rd_med_q3_autocorr_' + file_name + '_size_' + str(cache_size) + '_' + str(time.time()) + '.png')
        plt.close()

    t5 = time.time()
    print('Time for Median/Q3 Auto-Correlation: ', str(t5-t4)) 
    

    # Eviction Distance
    """ plt.figure(4)

    length = len(ev_x)
    ev_x = ev_x[int(0.05*length):int(0.95*length)]
    ev_y = ev_y[int(0.05*length):int(0.95*length)]

    plt.scatter(ev_x, ev_y, s=0.2)
    plt.title('Scatter Plot of Eviction Distance and Virtual Time')
    plt.xlabel('Virtual Time')
    plt.ylabel('Future Virtual Distance')
    plt.savefig('img/evict_dist_' + file_name + '_size_' + str(cache_size) + '.png')

    t5 = time.time()
    print('Time for Eviction Distance Plot: ', str(t5-t4)) """

    # Percentile Eviction Distance
    if 'ev' in write_modes:
        plt.figure(5)

        avg_ev_x = []
        p70_ev_y = []
        p80_ev_y = []
        p90_ev_y = []
        bucket_size = 100

        #temp_x_lst = []
        temp_y_lst = []
        for index, dist in enumerate(ev_y):
            if index % bucket_size == bucket_size - 1:
                avg_ev_x.append(ev_x[index])
                p70_ev_y.append(np.percentile(temp_y_lst, 70))
                p80_ev_y.append(np.percentile(temp_y_lst, 80))
                p90_ev_y.append(np.percentile(temp_y_lst, 90))
                #temp_x_lst = []
                temp_y_lst = []
            #temp_x_lst.append(ev_x[index])
            temp_y_lst.append(dist)

        p70_ev = plt.scatter(avg_ev_x, p70_ev_y, s=2, c='r', edgecolors='none')
        p80_ev = plt.scatter(avg_ev_x, p80_ev_y, s=2, c='g', edgecolors='none')
        p90_ev = plt.scatter(avg_ev_x, p90_ev_y, s=2, c='b', edgecolors='none')
        plt.title('Scatter Plot of Average Eviction Distance and Virtual Time')
        plt.xlabel('Virtual Time')
        plt.ylabel('Future Virtual Distance')
        plt.yscale('log')
        plt.legend((p70_ev, p80_ev, p90_ev), ('P70', 'P80', 'P90'), loc='upper right', markerscale=1.0)
        plt.savefig('img/evict_dist_stats_' + file_name + '_size_' + str(cache_size) + '_' + str(time.time()) + '.png')
        plt.close()

    t6 = time.time()
    print('Time for Average Eviction Distance Plot: ', str(t6-t5)) 
    
if __name__ == '__main__':
    main()



