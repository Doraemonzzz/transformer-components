
Transformer Components(tacos)

## 动机

现在有各种attention/transformer，但是由于实现细节不同，许多方法未必有效，本仓库的目的就是在严格控制变量的情况下测试各种attention的性能，测试任务初步定为：

- 单向语言模型ALM，例如GPT；
- 双向语言模型BLM，例如Bert/RoBerta；
- 视觉分类模型，例如Vit；



## 更新日志

- 2022/6/4: 初始化仓库，实现Vanilla Transformer；

删除需要手动删除egg-info。

## 规划
- 2022/6/6 ~ 2022/6/12: 确定传参方案, 迁移fairseq data preprocess, 跑通lm


## To DO

- 确定数据集；
- 跑通ALM, MLM；



## 参考资料

- [https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch)
- [https://github.com/lucidrains/x-transformers](https://github.com/lucidrains/x-transformers)
- [https://github.com/facebookresearch/fairseq](https://github.com/facebookresearch/fairseq)