# DeepfakeLab 生成配置的设置

## 已测试

1. 肤色同步 Color transfer to predicted face 目前选的是 rct
2. mask 校准 mask mode 选 learned-dst
3. Choose erode mask modifier ( -400..400 ) : 25 不知为何边缘有黑色区域
4. Choose blur mask modifier ( 0..400 ) : 25 不知为何边缘有黑线
5. 后续超清增加纹理似乎没啥用

## 待测试

1. 输入分辨率提高
2. 加入gan训练
3. 不同face type