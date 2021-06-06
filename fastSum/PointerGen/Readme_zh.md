# Pointer Generator

FastNLP实现论文：*[Get To The Point Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)*

原始代码：https://github.com/abisee/pointer-generator



## 输入数据格式
这里要求输入文件每一行都是一个json dict，dict需要有两个必要的key：'text'和'summary'，分别代表原文档和对应摘要，其值要求是list，保存已经分句结束的结果。

处理完后的例子：

- "text": ["london -lrb- cnn -rrb- a 19-year-old man was charged wednesday with terror offenses after he was arrested as he returned to britain from turkey , london 's metropolitan police said .", "yahya rashid , a uk national from northwest london , was detained at luton airport on tuesday after he arrived on a flight from istanbul , police said .", "he 's been charged with engaging in conduct in preparation of acts of terrorism , and with engaging in conduct with the intention of assisting others to commit acts of terrorism . both charges relate to the period between november 1 and march 31 .", "rashid is due to appear in westminster magistrates ' court on wednesday , police said .", "cnn 's lindsay isaac contributed to this report ."]
- "summary": ["london 's metropolitan police say the man was arrested at luton airport after landing on a flight from istanbul .", "he 's been charged with terror offenses allegedly committed since the start of november ."]



## 运行和使用

### 训练
训练Pointer Generator，命令行如下：
```shell
python train.py -train_data_path TRAIN_DATA_PATH -eval_data_path VALID_DATA_PATH -log_root LOG_ROOT_NAME -is_pointer_gen -is_coverage -n_epochs 33 -visible_gpu 0 -lr_coverage 0.025 -batch_size 16
```
其中`TRAIN_DATA_PATH`，`VALID_DATA_PATH`，`LOG_ROOT_NAME`分别代表训练数据路径，dev集数据路径以及模型存储的文件夹路径。如果要去除pointer机制或者coverage机制，可以去掉is_pointer_gen和is_coverage。



### 测试
```shell
python decode.py -decode_data_path TEST_DATA_PATH -train_data_path TRAIN_DATA_PATH -test_model CHECKPOINT -log_root LOG_ROOT_NAME -is_pointer_gen -is_coverage -test_data_name TEST_DATA_NAME -visible_gpu 0
```
其中`TEST_DATA_PATH`，`TRAIN_DATA_PATH`，`CHECKPOINT`，`LOG_ROOT_NAME`，`TEST_DATA_NAME` 分别代表测试数据路径，训练数据路径，测试模型路径，decode结果保存路径，测试数据集的名称。其中is_pointer_gen和is_coverage的加入要和训练模型时一致。

