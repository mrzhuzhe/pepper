# 单独 Efficientnet 分类

|  序号(version)   | 学习率  | 时间 | 分数（map） | backbone | epoch | batch | 优化器 | 备注 | 输入尺寸 |
|  ---- | ---- | ---- | ----- | ---- | ----- | ---- | ---- | ---- | ---- |
|  1(v6) | 1e-4 | 3h | 3.56 | effb3 + aux loss | 20 | 8 | adam | ---- | 512 |
|  2(v8) | 1e-3 | 2h | 3.63 | effb3 + aux los| 12 | 8 | adam | ---- | 512 |
|  3(v10) | 1e-3 | 2h | ----- | effb3 + aux los| 12 | 8 | madgrad | ---- | 512 |