import numpy as np
import random, math

lst = np.array(list(range(1,32)))
perms = set()
n = 10000

while len(perms) <= n:
    perm = tuple(np.random.permutation(lst))
    perms.add(perm)

def expected_val(permute, p):
    cur_lst = list(permute)
    
    for i in range(int(math.log(len(cur_lst), 2))):
        new_lst = []
        for j in range(int(len(cur_lst)/2)):
            small = min(cur_lst[2*j+1], cur_lst[2*j+2])
            big = max(cur_lst[2*j+1], cur_lst[2*j+2])
            if random.random() < p:
                new_lst.append(small)
            else:
                new_lst.append(big)
        cur_lst = new_lst
    return cur_lst[0]

sum = 0
for perm in perms:
    sum += expected_val(perm, 0.8)

print(sum / n)