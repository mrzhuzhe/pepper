#问题：
目前的识别都是把图片裁剪，变黑白，再用卷积神经网络vgg取“图案”, 但是你上次给的图片并没有明显的图案作为特征，只有纹理大小为特征，这个可能要预处理成一样大小，用纹理大小为特征？

#r0.01

##参考论文：动物皮图案纹理识别：
http://eprints.uad.ac.id/20034/1/Pretrained%20convolutional%20neural%20network%20for%20classification%20of%20tanning%20leather%20image.pdf

##数据：
1. tensorflow 材质纹理数据集 dtd：
https://www.tensorflow.org/datasets/catalog/dtd
2. 木头纹理数据集：
https://www.kaggle.com/edhenrivi/wood-samples?

##代码：
1. 钢材纹理分类的代码
https://www.kaggle.com/daisukelab/detector-steels-with-texture-for-sure-by-fast-ai
2. 基本服装图像分类：
https://www.tensorflow.org/tutorials/keras/classification

#r0.02

##数据集 - 
每个数据集都有一大堆以此数据集为基础的代码
1. 这个是损坏皮纹理的识别：皮被分为 好/中/坏/很差 几个档次，实际上是-离群距离-异常检测问题
https://www.kaggle.com/belkhirnacim/textiledefectdetection

2. 这个好像是布工艺品的识别，这个似乎没有人工标注分类
https://www.kaggle.com/gauravsingh69/texture-dataset?

3. 这个是语言描述和图案对应关系的数据集，例如 条纹 斑点纹 格子纹
https://www.kaggle.com/jmexpert/describable-textures-dataset-dtd?

4. 木头纹理 有标注是什么树 ， 不过这个图片有大有小（大部分有标注大小），需要预处理
https://www.kaggle.com/edhenrivi/wood-samples/notebooks


#其他
1. 发现网上肿瘤病变分类的免费数据集比较多