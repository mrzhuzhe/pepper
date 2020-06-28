import math
from itertools import combinations
from collections import Counter
"""
# 计算L2距离
def L(x, y, p=2):
    # x1 = [1, 1], x2 = [5,1]
    if len(x) == len(y) and len(x) > 1:
        sum = 0
        for i in range(len(x)):
            sum += math.pow(abs(x[i] - y[i]), p)
        return math.pow(sum, 1 / p)
    else:
        return 0

x1 = [1, 1]
x2 = [5, 1]
x3 = [4, 4]

# x1, x2 搞五次方干啥
for i in range(1, 5):
    r = {'1-{}'.format(c): L(x1, c, p=i) for c in [x2, x3]}
    #print(min(zip(r.values(), r.keys())))
print(r.values(), r.keys())
"""

arr = [(1, 1) , (2, 1), (3, 0)]
print(max(arr, key=lambda x: x[0]))
res = [k[-1] for k in arr]
print(Counter(res))