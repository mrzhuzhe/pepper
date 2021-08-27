# 单独 Efficientnet 分类

## Approach

- Work
1. eff-b3 + aux loss 做四分类 
2. map 的 trick 下对四分类全部有概率可以大幅提分
3. yolo 调参时 masic 和 mixup 对结果有极大的影响
4. 同样 yolo 反复提交重复结果也可以大幅提升map

- Not work
1. dropout 和 path dropout 似乎影响极小
2. 学习率似乎只跟数据和任务有关，跟自增关系不大
3. 不知为何我的梯度累积似乎也没起作用


## 实验记录

看起来fold2 4 似乎存在同一病人照片没分到同一组导致的leak

- Study 4 class

|  序号(version)   | 学习率  | 时间 | 分数（map） | backbone | epoch | batch | 优化器 | 备注 | 输入尺寸 |
|  ---- | ---- | ---- | ----- | ---- | ----- | ---- | ---- | ---- | ---- |
|  1(v6) | 1e-4 | 3h | 3.56 | effb3 + aux loss | 20 | 8 | adam | ---- | 512 |
|  2(v8) | 1e-3 | 2h | 3.63 | effb3 + aux los| 12 | 8 | adam | ---- | 512 |
|  3(v10) | 1e-3 | 2h | 3.72 | effb3 + aux los| 12 | 8 | madgrad | fold2 | 512 |
|  4(v11) | 1e-3 | 2h | 3.75 | effb3 + aux los| 12 | 8 | madgrad | 加入了wandb | 512 |
|  5(v12) | 1e-3 | 2h | 3.63 | effb3 + aux los| 12 | 8 | madgrad | fold0 | 512 |
|  5(v12) | 1e-3 | 2h | 3.5 | effb3 + aux los| 12 | 8 | madgrad | fold1 | 512 |
|  5(v12) | 1e-3 | 2h | 3.65 | effb3 + aux los| 12 | 8 | madgrad | fold3 | 512 |
|  5(v12) | 1e-3 | 2h | 3.73 | effb3 + aux los| 12 | 8 | madgrad | fold4 | 512 |
|  6 | 1e-3 + 1e-4 | 2h | 3.76 | effb3 + aux los| 12 | 8 | madgrad | fold0 | 512 |
|  6 | 1e-3 + 1e-4 | 2h | 3.92 | effb3 + aux los| 12 | 8 | madgrad | fold1 似乎不对 | 512 |
|  6 | 1e-3 + 1e-4 | 2h | 3.85  | effb3 + aux los| 12 | 8 | madgrad | fold2 | 512 |
|  6 | 1e-3 + 1e-4 | 2h | 3.7 | effb3 + aux los| 12 | 8 | madgrad | fold3 | 512 |
|  6 | 1e-3 + 1e-4 | 2h | 3.6 | effb3 + aux los| 12 | 8 | madgrad | fold4 似乎不对 | 512 |
|  7 | 1e-4 | 1h | 3.54 | effb3 + aux los 加深为四层 | 12 | 18 | madgrad | fold3  | 512 |
|  8 | 1e-4 | 1h | 3.51 | effb3 + aux los | 12 | 8 | madgrad | fold0 opacity 和 none 跟着一起softmax输出  | 512 |
|  9 | 1e-4 + 1e-5 | 2h | 3.52 | effb3 + aux los | 12 | 8 | madgrad | fold0 opacity 和 none 跟着一起softmax输出  | 512 |
|  10 | 1e-3 | 1h | 3.33 | effb3 + aux los  | 12 | 8 | madgrad | fold0 opacity 和 none 跟着一起softmax输出  | 512 |
|  11 | 1e-3 | 1h | .5703 | effb3 + aux los  | 12 | 8 | madgrad | fold0 只计算 opacity 和 none 的loss  | 512 |
|  12 | 1e-4 | 1h | .5711 | effb3 + aux los  | 12 | 8 | madgrad | fold0 只计算 opacity 和 none 的loss | 512 |
|  13 | 1e-4 | 1h | 3.55 | effv2 + aux los  | 12 | 8 | madgrad | 还原 | 512 |
|  14 | 1e-5 | 1h | 3.63 | effv2 + aux los  | 12 | 8 | madgrad | ---- | 512 |
|  15 | 1e-5 | 1h | 3.62 | effv2 + aux los + 0.5 dropout | 12 | 8 | madgrad | ---- | 512 |
|  16 | 1e-4 | 1h | na | effv2 + aux los | 12 | 8 | madgrad | 以512训练但是以640infer反而从.444掉到了.436 | 640 |
|  17 | 1e-3 + 1e-4 | 3h | 0.395 | effv2 + aux los + 0.5 dropout + 0.4 path dropout + hard augment | 25 | 16 | madgrad | fold0 | 512 |
|  17 | 1e-3 + 1e-4 | 3h | 0.395 | effv2 + aux los + 0.5 dropout + 0.4 path dropout + hard augment | 25 | 16 | madgrad | fold1 | 512 |
|  17 | 1e-3 + 1e-4 | 3h | 0.384 | effv2 + aux los + 0.5 dropout + 0.4 path dropout + hard augment | 25 | 16 | madgrad | fold2 | 512 |
|  17 | 1e-3 + 1e-4 | 3h | 0.381 | effv2 + aux los + 0.5 dropout + 0.4 path dropout + hard augment | 25 | 16 | madgrad | fold3 | 512 |
|  17 | 1e-3 + 1e-4 | 3h | 0.366 | effv2 + aux los + 0.5 dropout + 0.4 path dropout + hard augment | 25 | 16 | madgrad | fold4 | 512 |

- 2class 这个过拟合非常严重

- yolo 

没有augment时过拟合极其严重，有augment时mixup影响又过大

|  序号(version)   | 学习率  | 时间 | 分数（map） | backbone | epoch | batch | 优化器 | 备注 | 输入尺寸 |
|  ---- | ---- | ---- | ----- | ---- | ----- | ---- | ---- | ---- | ---- |
|  1 | ---- | 18m | .483 | s | 10 | ---- | ---- | ---- | 512 |
|  2 | ---- | 20m | .510 | m | 10 | ---- | ---- | ---- | 512 |
|  3 | ---- | 30m | .430 | m | 10 | ---- | ---- | ---- | 640 |
|  4 | ---- | 30m | .450 | x | 10 | ---- | ---- | ---- | 512 |
|  5 | ---- | 1.5h | .450 | x | 19 | ---- | ---- | 没有mixup | 512 |
|  6 | evlvo | 很多小时 | 最高 .425 | m | ---- | ---- | 跑evlvo NAS算法 |  | 512 |
|  7 | ---- | 30m | .450 | m | 20 | 24 | ---- | 只有.2的mixup | 512 |
|  8 | ---- | 2h | .487 | x | 30 | 24 | ---- | 强augment mosac .7 加 .16mixup | 512 |
|  9 | ---- | 2h | .485 | x | 30 | 24 | ---- | 更强augment mosac 1 无mixup | 512 |
|  10 | ---- | 2h | .47 | x | 30 | 24 | ---- | 更强augment mosac 1 加 无mixup | 512 |

yolom 最终：
score: (0.4947 + 0.5103 + 0.4848 +0.4692 +0.5198)/5 avg 0.49575 LB: 0.142 tta 0.145

yolox 最终：
score: (0.481 + 0.488 + 0.525 + 0.525 + 0.488) 应该是 0.148