import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PyMimircache.cacheReader.csvReader import CsvReader
from PyMimircache.cache.optimal import Optimal

init_params = {'label': 2}
file_name = '29_sample'
reader = CsvReader('/research/george/ranktest/features/' + file_name + '.csv',
    init_params=init_params)

cache_size = 500
rand_factor = 0.1
opt = Optimal(cache_size, reader)

# Timestamp vector
fut_rd_arr1 = []
# Future Reuse vector
fut_rd_arr2 = []

# Dict to store the last vtime for a given id
#id_to_vtime = {}

for request in reader:
    opt.access(request)
    #id_to_vtime[request] = opt.ts
    for req_id in opt.pq:
        if random.random() < rand_factor:
            fut_rd_arr1.append(opt.ts)
            fut_rd_arr2.append(-opt.pq[req_id] - opt.ts)
    if (opt.ts % 10000 == 0):
        print('progress', opt.ts)

x = fut_rd_arr1 # virtual time
y = fut_rd_arr2 # future reuse distance

# Scatter plot of virtual time and future reuse distance    
plt.figure(1)
plt.scatter(x, y, s=0.2)
plt.savefig('img/scatter_' + file_name + '_size_' + str(cache_size) + '.png')


# 2d Histogram
plt.figure(2)
plt.hist2d(x, y, bins=[opt.ts//100, 20], range=[[0, opt.ts], [0, cache_size*10]])
plt.savefig('img/hist_' + file_name + '_size_' + str(cache_size) + '.png')

