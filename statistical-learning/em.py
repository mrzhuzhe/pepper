"""
http://fangs.in/post/thinkstats/likelihood/
https://github.com/fengdu78/lihang-code/blob/master/%E7%AC%AC09%E7%AB%A0%20EM%E7%AE%97%E6%B3%95%E5%8F%8A%E5%85%B6%E6%8E%A8%E5%B9%BF/9.Expectation_Maximization.ipynb
怎么没有q函数相关的呢
"""

import numpy as np
import math

# 初始值
pro_A, pro_B, por_C = 0.5, 0.5, 0.5

#	概率密度函数
def pmf(i, pro_A, pro_B, por_C):
    pro_1 = pro_A * math.pow(pro_B, data[i]) * math.pow(
        (1 - pro_B), 1 - data[i])
    pro_2 = pro_A * math.pow(pro_C, data[i]) * math.pow(
        (1 - pro_C), 1 - data[i])
    return pro_1 / (pro_1 + pro_2)

class EM:
    def __init__(self, prob):
        self.pro_A, self.pro_B, self.pro_C = prob

    # e_step
    def pmf(self, i):
        pro_1 = self.pro_A * math.pow(self.pro_B, data[i]) * math.pow(
            (1 - self.pro_B), 1 - data[i])
        pro_2 = (1 - self.pro_A) * math.pow(self.pro_C, data[i]) * math.pow(
            (1 - self.pro_C), 1 - data[i])
        return pro_1 / (pro_1 + pro_2)

    # m_step
    def fit(self, data):
        count = len(data)
        print('init prob:{}, {}, {}'.format(self.pro_A, self.pro_B,
                                            self.pro_C))
        for d in range(count):
            _ = yield
            # e 步骤
            _pmf = [self.pmf(k) for k in range(count)]
            # m 步
            pro_A = 1 / count * sum(_pmf)
            #   根据 data 做了最大似然 TODO这里手推一下
            """
            第一轮 时
             pro_b = 0.5 * 6 / 0.5 * 10
             pro_c = (1 - 0.5) * 6 / (1 - 0.5) * 10
            第二轮 时
             以第一轮为基准
             _pmf = 0.5
             下面两个还是不变
             pro_b = 0.5 * 6 / 0.5 * 10
             pro_c = (1 - 0.5) * 6 / (1 - 0.5) * 10
            """
            pro_B = sum([_pmf[k] * data[k] for k in range(count)]) / sum(
                [_pmf[k] for k in range(count)])
            pro_C = sum([(1 - _pmf[k]) * data[k]
                         for k in range(count)]) / sum([(1 - _pmf[k])
                                                        for k in range(count)])
            print('{}/{}  pro_a:{:.3f}, pro_b:{:.3f}, pro_c:{:.3f}'.format(
                d + 1, count, pro_A, pro_B, pro_C))
            self.pro_A = pro_A
            self.pro_B = pro_B
            self.pro_C = pro_C

data=[1,1,0,1,0,0,1,0,1,1]

em = EM(prob=[0.5, 0.5, 0.5])
f = em.fit(data)
next(f)

# 第一次迭代
f.send(1)

# 第二次 generator 函数 https://docs.python.org/2.5/whatsnew/pep-342.html
f.send(2)


em = EM(prob=[0.4, 0.6, 0.7])
f2 = em.fit(data)
next(f2)


f2.send(1)

f2.send(2)
