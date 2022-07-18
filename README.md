
Transformer Components(tacos)

## 动机

现在有各种attention/transformer，但是由于实现细节不同，许多方法未必有效，本仓库的目的就是在严格控制变量的情况下测试各种attention的性能，测试任务初步定为：

- 单向语言模型ALM，例如GPT；
- 双向语言模型BLM，例如Bert/RoBerta；
- 视觉分类模型，例如Vit；



## 更新日志

- 2022/7/4~2022/7/10: 
  - [x] 完成transformer基本部件的实现；
  - [x] 完成pypi上传；
- 2022/7/11~2022/7/17：
  - [x] 实现各个版本的norm；



## 规划
- 2022/7/18~2022/7/24：
  - [ ] 完成代码自动release；
  - [ ] 完成英文版readme；
  - [ ] 实现linear attention；
  - [ ] 实现测试代码；
  - [ ] 完成tacos中的transformer部分，上周只完成的部件的实现；



## 参考资料

- [https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch)
- [https://github.com/lucidrains/x-transformers](https://github.com/lucidrains/x-transformers)
- [https://github.com/facebookresearch/fairseq](https://github.com/facebookresearch/fairseq)