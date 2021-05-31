# Baseline

使用FastNLP实现的以Transformer、DeepLSTM（二者可选其一）作为编码器，序列标注作为解码器进行抽取式摘要的代码，可作为抽取式摘要研究的基线。



## 环境安装

### 最新的FastNLP

```shell
pip install git+https://gitee.com/fastnlp/fastNLP@dev
```



### PyRouge

为了获得正确的Rouge评分，建议使用以下命令安装Rouge环境：

```shell
sudo apt-get install libxml-perl libxml-dom-perl
pip install git+git://github.com/bheinzerling/pyrouge
export PYROUGE_HOME_DIR=the/path/to/RELEASE-1.5.5
pyrouge_set_rouge_path $PYROUGE_HOME_DIR
chmod +x $PYROUGE_HOME_DIR/ROUGE-1.5.5.pl
```

对于RELEASE-1.5.5，请务必在RELEASE-1.5.5/data中构建Wordnet 2.0而不是Wordnet 1.6，你可以参考此[链接](https://github.com/andersjo/pyrouge/tree/master/tools/ROUGE-1.5.5)中的说明

```shell
cd $PYROUGE_HOME_DIR/data/WordNet-2.0-Exceptions/
./buildExeptionDB.pl . exc WordNet-2.0.exc.db
cd ../
ln -s WordNet-2.0-Exceptions/WordNet-2.0.exc.db WordNet-2.0.exc.db
```

**测试ROUGE是否安装正确**

```shell
pyrouge_set_rouge_path /absolute/path/to/ROUGE-1.5.5/directory
python -m pyrouge.test
```

安装pyrouge：

```shell
pip install pyrouge
```



## 数据集和词典

1. 可以参考testdata文件夹中的数据样例，数据集均以jsonl格式存储；
2. 词典推荐使用[glove.42B.300d](https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/gluon/embeddings/glove/glove.42B.300d.zip)



## 修改ROUGE路径

**在tools中的untils.py文件中写入ROUGE安装的绝对路径**

```python
############## 修改成自己的ROUGE路径 ##############
ROUGE_PATH = "~/ROUGE/"
```



## 运行和使用

LSTM + Sequence Labeling:

```shell
python train.py --cuda --gpu <gpuid> --sentence_encoder deeplstm --sentence_decoder SeqLab --save_root <savedir> --log_root <logdir> --lr_descent --grad_clip --max_grad_norm 10 --use_pyrouge
```

Transformer + Sequence Labeling:

```shell
python train.py --cuda --gpu <gpuid> --sentence_encoder transformer --sentence_decoder SeqLab --save_root <savedir> --log_root <logdir> --lr_descent --grad_clip --max_grad_norm 10 --use_pyrouge
```

