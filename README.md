# MICCAI-Myopic-Maculopathy-Analysis
MICCAI 2023近视性黄斑病变竞赛

我们的模型框架图为

![avatar](/picture_1.png)

## 第一阶段（step1）
Swin transformer的训练代码为`train_swin.py`，dataloader的代码为`dataloader_swin.py`

Auto-encoder的训练代码见`cann-for-hc.ipynb`

Resnet-50的训练代码见`run.py`。

运行时注意修改其中的模型路径。


## 第二阶段（step2）
训练代码见`all_train_2.py`，测试代码见`all_test.py`，dataloader的代码为`dataloader_all.py`
