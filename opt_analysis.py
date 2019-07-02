import numpy as np
import random, sys, time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PyMimircache.cacheReader.csvReader import CsvReader
from PyMimircache.cache.optimal import Optimal

init_params = {'label': 2}
file_name = sys.argv[1] # Command Line Argument
reader = CsvReader('/research/george/ranktest/features/' + file_name + '.csv',
    init_params=init_params)

cache_size = 500
rand_factor = 0.1
opt = Optimal(cache_size, reader)

# Timestamp values
fut_rd_arr1 = []
# Future Reuse values
fut_rd_arr2 = []

# Median values
med_x = []
med_y = []

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
    for req_id in opt.pq:
        if random.random() < rand_factor:
            fut_rd_arr1.append(opt.ts)
            fut_rd_arr2.append(-opt.pq[req_id] - opt.ts)
            temp_lst.append(-opt.pq[req_id] - opt.ts) 
    if temp_lst:
        med_x.append(opt.ts)
        med_y.append(np.median(temp_lst))
    if ev_item:
        ev_x.append(opt.ts)
        ev_y.append(opt.ts - id_to_vtime[ev_item[0]])
    if (opt.ts % 100000 == 0):
        print('progress', opt.ts)

t1 = time.time()  
print('Time for Iteration: ', str(t1-t0))

x = fut_rd_arr1 # virtual time
y = fut_rd_arr2 # future reuse distance

# Scatter plot of virtual time and future reuse distance
plt.figure(1)

samples = np.random.randint(0, len(x), size=int(0.1*len(x))) # Samples 10% of values to plot
x_sample = [x[i] for i in samples]
y_sample = [y[i] for i in samples]

plt.scatter(x_sample, y_sample, s=0.2)
plt.title('Scatter Plot of Future Reuse Distances and Virtual Time')
plt.xlabel('Virtual Time')
plt.ylabel('Future Virtual Distance')
plt.savefig('img/scatter_' + file_name + '_size_' + str(cache_size) + '.png')

t2 = time.time()
print('Time for Scatter Plot: ', str(t2-t1))

# 2D Histogram
plt.figure(2)

plt.hist2d(x, y, bins=[opt.ts//100, 25], range=[[0, opt.ts], [0, cache_size*10]])
plt.title('2D Histogram of Future Reuse Distance Frequency and Virtual Time')
plt.xlabel('Virtual Time')
plt.ylabel('Future Virtual Distance')
plt.savefig('img/hist_' + file_name + '_size_' + str(cache_size) + '.png')

t3 = time.time()
print('Time for 2D Histogram: ', str(t3-t2))

# Median Future Reuse Distance
plt.figure(3)

length = len(med_x)
med_x = med_x[int(0.05*length):int(0.95*length)]
med_y = med_y[int(0.05*length):int(0.95*length)]

plt.scatter(med_x, med_y, s=0.2)
plt.title('Scatter Plot of Future Reuse Distance Median and Virtual Time')
plt.xlabel('Virtual Time')
plt.ylabel('Future Virtual Distance')
plt.savefig('img/median_' + file_name + '_size_' + str(cache_size) + '.png')

t4 = time.time()
print('Time for Median Plot: ', str(t4-t3))

# Eviction Distance
plt.figure(4)

length = len(ev_x)
ev_x = ev_x[int(0.05*length):int(0.95*length)]
ev_y = ev_y[int(0.05*length):int(0.95*length)]

plt.scatter(ev_x, ev_y, s=0.2)
plt.title('Scatter Plot of Eviction Distance and Virtual Time')
plt.xlabel('Virtual Time')
plt.ylabel('Future Virtual Distance')
plt.savefig('img/evict_dist_' + file_name + '_size_' + str(cache_size) + '.png')

t5 = time.time()
print('Time for Eviction Distance Plot: ', str(t5-t4))

# Average Eviction Distance
plt.figure(5)

avg_ev_x = []
avg_ev_y = []
bucket_size = 100

count = 0
temp_x_lst = []
temp_y_lst = []
for index, dist in enumerate(ev_y):
    if count == bucket_size:
        avg_ev_x.append(temp_x_lst[-1])
        avg_ev_y.append(np.mean(temp_y_lst))
        count = 0
        temp_x_lst = []
        temp_y_lst = []
    temp_x_lst.append(ev_x[index])
    temp_y_lst.append(dist)
    count += 1

plt.scatter(avg_ev_x, avg_ev_y, s=0.2)
plt.title('Scatter Plot of Average Eviction Distance and Virtual Time')
plt.xlabel('Virtual Time')
plt.ylabel('Future Virtual Distance')
plt.savefig('img/evict_dist_avg_' + file_name + '_size_' + str(cache_size) + '.png')

t6 = time.time()
print('Time for Average Eviction Distance Plot: ', str(t6-t5))
    
        



