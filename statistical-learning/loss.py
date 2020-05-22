
import numpy as np
import math
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10,8))
x = np.linspace(start=-1, stop=2, num=1001, dtype=np.float)
logi = np.log(1 + np.exp(-x)) / math.log(2)
boost = np.exp(-x)
y_01 = x < 0
y_hinge = 1.0 - x
y_hinge[y_hinge < 0] = 0

plt.plot(x, y_01, 'g-', mec='k', label='（0/1损失）0/1 Loss', lw=2)
plt.plot(x, y_hinge, 'b-', mec='k', label='（合页损失）Hinge Loss', lw=2)
plt.plot(x, boost, 'm--', mec='k', label='（指数损失）Adaboost Loss', lw=2)
plt.plot(x, logi, 'r-', mec='k', label='（逻辑斯谛损失）Logistic Loss', lw=2)
plt.grid(True, ls='--')
plt.legend(loc='upper right',fontsize=15)
plt.xlabel('函数间隔:$yf(x)$',fontsize=20)
plt.title('损失函数',fontsize=20)
plt.show()
