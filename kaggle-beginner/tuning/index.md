# 如何调参数

## 重要程度

1. 在众多参数中 batchsize和学习率是最重要的
2. 其次是epoch数量 earlystop reduce等
3. 数据自增等几乎没有影响

## 实践

1. batchsize 会提升速度，听说也能提高范化性能
2. learning rate 太大会造成次优解 太小会收敛很慢
3. epoch 似乎大 batchsize 会需要更多epoch ？ 
4. tensorlow 的 model callback  learning rate schedule


## 待查

1. batchsize epoch learning_rate 之间的关系？
2. tensorflow 的常见设置 callback 日志输出？
3. efficientnet 容量 对结果的影响
4. 如何根据学习曲线 确定次优解
5. tpu tensorboard
6. 试一下loss函数

