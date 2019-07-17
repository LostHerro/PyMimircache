import time, sys

scan_size = int(sys.argv[1])
num_scans = int(sys.argv[2])

t = int(time.time())
f = open('traces/scan_' + str(scan_size) + '_' + str(num_scans), 'w+')

for i in range(num_scans):
    for j in range(scan_size):
        f.write(str(t) + ' ' + str(j) + '\n')
        t += 1
        
