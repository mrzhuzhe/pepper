# pepper
 a collection of implements for cs224n 2019 homework

> 用来存放 CS224n 2019 的作业

## install
``` python
  conda install --file requirements.txt
```

## note
 -  conda 源需要指向tuna源
 -  [a1]  需要查一下 trunstSVD 有点忘掉了 目前似乎是把高维度对称矩阵pca降维
 -  [a2]  需要看一下 neg-sample 梯度是怎么计算的（网上搜到的loss是对的梯度是错的）
 -  [pytorchTest]  因为assgin3开始 需要pytorch 基础，所以现在改为现看pytorch的nlp教程

# on going
 -  [lstm] textProccess/lstm 

### Script
``` python
# pytorch 入门教程 多 gpu分配 和多核需要再看看
python pytorchTest/vision.py

```


## resouce & meterial refference
 - some solutions https://github.com/ZacBi/CS224n-2019-solutions
 - cource materials https://github.com/zhanlaoban/CS224N-Stanford-Winter-2019
 - some notes & solutions https://blog.csdn.net/cindy_1102/article/category/8804737
