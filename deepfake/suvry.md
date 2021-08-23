# Deepfake 调查

- 问题

1. 为什么默认没有用gan？？？？ 说实话这个训练不是gan么？

- faceswap 和 deepfakelab的 差别

1. faceswap 用的是 dl 架构 dfl 用的是 liae 架构
2. dfl 核心依赖于预先生成 mask 产生 人脸配准 aligned 而 faceswap 只用脸部的一小块
3. dfl 在生成时针对肤色 边缘模糊 做了更多后期处理