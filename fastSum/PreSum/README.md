# PreSum
FastNLP实现的EMNLP2019论文 [Text Summarization with Pretrained Encoders](https://arxiv.org/pdf/1908.08345)
<br>
原始代码[地址](https://github.com/nlpyang/PreSumm)


## 包依赖
- Python 3.7
- [PyTorch](https://github.com/pytorch/pytorch) >= 1.0.1
- [fastNLP](https://github.com/fastnlp/fastNLP) 0.6.0
- [pyrouge](https://github.com/bheinzerling/pyrouge) 0.1.3
- [rouge](https://github.com/pltrdy/rouge) 1.0.0
- [transformers](https://github.com/huggingface/transformers) >= 1.2.0 



## 数据预处理
调用preprocess.py来预处理数据。示例如下：
```
python preprocess.py --raw_path INPUT_PATH --save_path data/OUTPUT_PATH --log_file LOG_PATH
```
其中INPUT_PATH，OUTPUT_PATH和LOG_PATH分别代表预处理的输入目录，输出目录名字以及log的路径。
这里要求INPUT_PATH目录下有xx.train.jsonl， xx.val.jsonl， xx.test.jsonl 命名的文件，同时要求输入文件每一行都是一个json dict，dict需要有两个必要的key：'text'和'summary'，分别代表原文档和对应摘要，其值要求是list，保存已经分句结束的结果。



## TransformerABS
训练transformer为基础结构的生成式模型，命令行如下：
```
python train_presum.py -task abs -mode train -dec_dropout 0.2 -save_path SAVE_DIR -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 16 -accum_count 5 -use_bert_emb true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 0,1,2,3 -log_file LOG_PATH -label_type INPUT_DIR -valid_steps 2000 -n_epochs 10 -max_summary_len 600
```
其中SAVE_DIR, LOG_PATH和INPUT_DIR分别代表了存储训练模型的目录，log的路径以及输入数据的文件夹名字。特别的，INPUT_DIR对应了数据预处理部分的OUTPUT_PATH，只需要告知data目录下输入的文件夹名字即可。
<br>

测试transformer为基础结构的生成式模型，命令行如下：
```
python train_presum.py -task abs -mode test -test_batch_size 12 -log_file LOG_PATH -test_from CHECKPOINT_PATH -sep_optim true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -block_trigram True -label_type INPUT_DIR -max_summary_len 600 -decode_path DECODE_PATH
```
其中LOG_PATH，CHECKPOINT_PATH，INPUT_DIR，DECODE_PATH分别代表了log存储路径，测试模型的路径，包含测试集的文件夹名字（对应了数据预处理部分的OUTPUT_PATH），以及保存decode结果的路径名字。max_length ，min_length 和alpha 需要根据不同数据集特性调节。



## BERTSUMABS
训练BERT为基础结构的生成式模型，命令行如下：
```
python train_presum.py -task abs -mode train -dec_dropout 0.1 -save_path SAVE_DIR  -sep_optim False -lr 0.05 -save_checkpoint_steps 2000 -batch_size 8 -accum_count 8 -use_bert_emb true -warmup_steps 10000 -max_pos 512 -visible_gpus 0,1,2,3 -log_file LOG_PATH -label_type INPUT_DIR -valid_steps 2000 -n_epochs 10 -max_summary_len 600 -encoder baseline -enc_dropout 0.1 -enc_hidden_size 512  -enc_layers 6 -enc_ff_size 2048 -dec_layers 6 -dec_hidden_size 512 -dec_ff_size 2048
```
其中SAVE_DIR, LOG_PATH和INPUT_DIR分别代表了存储训练模型的目录，log的路径以及输入数据的文件夹名字。特别的，INPUT_DIR对应了数据预处理部分的OUTPUT_PATH，只需要告知data目录下输入的文件夹名字即可。


测试BERT为基础结构的生成式模型，命令行如下：
```
python train_presum.py -task abs -mode test -test_batch_size 12 -log_file LOG_PATH -test_from CHECKPOINT_PATH -sep_optim False -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -block_trigram True -label_type INPUT_DIR -max_summary_len 300 -decode_path DECODE_PATH -encoder baseline
```
其中LOG_PATH，CHECKPOINT_PATH，INPUT_DIR，DECODE_PATH分别代表了log存储路径，测试模型的路径，包含测试集的文件夹名字（对应了数据预处理部分的OUTPUT_PATH），以及保存decode结果的路径名字。max_length ，min_length 和alpha 需要根据不同数据集特性调节。

## TransformerEXT
训练transformer为基础结构的抽取式模型，命令行如下：
测试transformer为基础结构的抽取式模型，命令行如下：



## BERTSUMEXT
训练BERT为基础结构的抽取式模型，命令行如下：
测试BERT为基础结构的抽取式模型，命令行如下：
