# pepper
 a collection of implements for cs224n 2019 homework
 a collection of pytorch tutorial
 and bert demo

> 用来存放 CS224n 2019 的作业 & pytorch 入门教程
> 记录笔记 总结

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

### pytorch 入门

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

### assign

``` python

# shift left-arc right-arc
python assign03/parser_transitions.py part_d

"""

 输入数据 样例：
  train_set
   {
      'word':
       ['in', 'an', 'oct.', '19', 'review', 'of', '``', 'the', 'misanthrope', "''", 'at', 'chicago', "'s", 'goodman', 'theatre', '-lrb-', '``', 'revitalized', 'classics', 'take', 'the', 'stage', 'in', 'windy', 'city', ',', "''", 'leisure', '&', 'arts', '-rrb-', ',', 'the', 'role', 'of', 'celimene', ',', 'played', 'by', 'kim', 'cattrall', ',', 'was', 'mistakenly', 'attributed', 'to', 'christina', 'haag', '.'],
     'pos':
       ['IN', 'DT', 'NNP', 'CD', 'NN', 'IN', '``', 'DT', 'NN', "''", 'IN', 'NNP', 'POS', 'NNP', 'NNP', '-LRB-', '``', 'VBN', 'NNS', 'VB', 'DT', 'NN', 'IN', 'NNP', 'NNP', ',', "''", 'NN', 'CC', 'NNS', '-RRB-', ',', 'DT', 'NN', 'IN', 'NNP', ',', 'VBN', 'IN', 'NNP', 'NNP', ',', 'VBD', 'RB', 'VBN', 'TO', 'NNP', 'NNP', '.'],
     'head':
       [5, 5, 5, 5, 45, 9, 9, 9, 5, 9, 15, 15, 12, 15, 9, 20, 20, 19, 20, 5, 22, 20, 25, 25, 20, 20, 20, 20, 28, 28, 20, 45, 34, 45, 36, 34, 34, 34, 41, 41, 38, 34, 45, 45, 0, 48, 48, 45, 45],
     'label':
        ['case', 'det', 'compound', 'nummod', 'nmod', 'case', 'punct', 'det', 'nmod', 'punct', 'case', 'nmod:poss', 'case', 'compound', 'nmod', 'punct', 'punct', 'amod', 'nsubj', 'dep', 'det', 'dobj', 'case', 'compound', 'nmod', 'punct', 'punct', 'dep', 'cc', 'conj', 'punct', 'punct', 'det', 'nsubjpass', 'case', 'nmod', 'punct', 'acl', 'case', 'compound', 'nmod', 'punct', 'auxpass', 'advmod', 'root', 'case', 'compound', 'nmod', 'punct']
    }
 wordembedding 样例：
   accelerators
     0.181292 -0.667547 -1.56269 1.07374 -0.578965 -1.48894 -3.40869 -0.995061 0.00719613 1.59514 0.666047 -1.23074 -1.07743 -0.151945 -0.788508 0.871682 1.44595 0.0136208 -1.34467 1.1571 0.130709 0.0227585 0.282243 0.229792 -0.0815991 -0.0376202 0.428753 0.303739 -0.73318 -0.557974 0.508922 -0.458103 -0.309525 -0.841847 1.36923 -1.28841 -1.65283 -0.621058 -0.869718 1.90532 0.00530639 -1.19798 0.830816 -1.04491 0.519946 0.066836 0.613915 -0.331479 -0.473813 -0.767639
  目标函数：forward 结果 logist 和 train_y 的 error
  损失计算过程：下一步操作的denpendence 和实际的 偏差（ TODO 重点 ）
  损失：交叉熵

"""
python assign03/run.py


```


## 中文 nlp 实练

### colab 训练
1. colab 中bert fine-turn 简单demo 数据只有500条   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pP83paixvnu8fm8Xv69RwD831pBOZnPg)
2. bert最简单的文章相关性判定 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VDCsK0M-eUjWKb_cz5rojw3FnjcNJvsp#scrollTo=n9AT5FTTyIep)
3. mnt transformer 英翻中 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FJ1XQ18PTsu2jGV282mCbPKXKvroemYL)

### 笔记
1. 本地安装需要 tersorflow 1.11 py3.6
2. 为了训练方便，改为colab notebook

## resouce & meterial refference
 - some solutions https://github.com/ZacBi/CS224n-2019-solutions
 - cource materials https://github.com/zhanlaoban/CS224N-Stanford-Winter-2019
 - some notes & solutions https://blog.csdn.net/cindy_1102/article/category/8804737
 - amazing blog form Mr Lee Meng https://leemeng.tw/neural-machine-translation-with-transformer-and-tensorflow2.html
