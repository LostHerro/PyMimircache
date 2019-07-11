#import PyMimircache as m
import sys

file_name = sys.argv[1]
time_ind = int(sys.argv[2])
id_ind = int(sys.argv[3])
count = 0

with open(file_name) as f:
    g = open(file_name + '_no_feat.csv', 'w+')
    for line in f:
        ln = line.split()
        g.write(ln[time_ind] + ',' + ln[id_ind] + '\n')
        if count == 1000000:
            print('asdf')
            count = 0
        count += 1
    g.close()
        

