# pepper
 a collection of implements for cs224n 2019 homework
 a collection of pytorch tutorial
 and bert demo

> 用来存放 CS224n 2019 的作业 & pytorch 入门教程

## install
``` python
  conda install --file requirements.txt
```

## note
 -  conda 源需要指向tuna源
 -  [a1]  需要查一下 trunstSVD 有点忘掉了 目前似乎是把高维度对称矩阵pca降维
 -  [a2]  需要看一下 neg-sample 梯度是怎么计算的（网上搜到的loss是对的梯度是错的）
 -  [pytorchTest]  因为assgin3开始 需要pytorch 基础，所以现在改为现看pytorch的nlp教程


## Script
``` python

# pytorch 入门教程 多 gpu分配 和多核需要再看看
python pytorchTest/vision.py

# 基本的 rnn 用了 i2o i2h（这是模仿lstm?） 和一个隐藏层  2 隐藏层
python textProccess/rnn/main.py

# char-generate 的 rnn 用了 i2o i2h 020  隐藏层 2 隐藏层  output 2 隐藏层  2 input
python textProccess/generate/main.py

# torchtext 判别器， 查看并更改了源码解决了torchtext 每次都要下载数据的问题
python torchText/classify/main.py

# TODO 完全用来学习的 transformer language model
python torchText/transformer/main.py

```

## bert 中的一些经验记录

### 例子
1. colab 中bert fine-turn 简单demo 数据只有500条   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pP83paixvnu8fm8Xv69RwD831pBOZnPg)

### 笔记
1. 本地安装需要 tersorflow 1.11 py3.6
2. 为了测试方便，改为colab notebook 见 /bertnore 目录

## resouce & meterial refference
 - some solutions https://github.com/ZacBi/CS224n-2019-solutions
 - cource materials https://github.com/zhanlaoban/CS224N-Stanford-Winter-2019
 - some notes & solutions https://blog.csdn.net/cindy_1102/article/category/8804737
