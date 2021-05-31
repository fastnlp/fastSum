# Baseline

In this folder, we provide the code (implemented by FastNLP) that uses transformer or deep LSTM as encoder and sequence labeling as decoder for extractive summarization, which can be used as the baseline of extractive summarization research.



## Environment

### Latest FastNLP

```shell
pip install git+https://gitee.com/fastnlp/fastNLP@dev
```



### PyRouge

In order to get correct ROUGE scores, we recommend using the following commands to install the ROUGE environment:

```shell
sudo apt-get install libxml-perl libxml-dom-perl
pip install git+git://github.com/bheinzerling/pyrouge
export PYROUGE_HOME_DIR=the/path/to/RELEASE-1.5.5
pyrouge_set_rouge_path $PYROUGE_HOME_DIR
chmod +x $PYROUGE_HOME_DIR/ROUGE-1.5.5.pl
```

You can refer to https://github.com/andersjo/pyrouge/tree/master/tools/ROUGE-1.5.5 for RELEASE-1.5.5. Remember to build Wordnet 2.0 instead of 1.6 in RELEASE-1.5.5/data.

```shell
cd $PYROUGE_HOME_DIR/data/WordNet-2.0-Exceptions/
./buildExeptionDB.pl . exc WordNet-2.0.exc.db
cd ../
ln -s WordNet-2.0-Exceptions/WordNet-2.0.exc.db WordNet-2.0.exc.db
```

**Test whether the installation of rouge is successful:**

```shell
pyrouge_set_rouge_path /absolute/path/to/ROUGE-1.5.5/directory
python -m pyrouge.test
```

Install PyRouge:

```shell
pip install pyrouge
```



## Dataset and Vob

1. You can refer to the data samples in testdata folder, which are stored in jsonl format.
2. [Glove.42b.300d](https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/gluon/embeddings/glove/glove.42B.300d.zip) dictionary is recommended.



## Modify the ROUGE path

**Write the absolute path of the ROUGE installation in the tools\untils.py ! **

```python
############## Modify to your own Rouge path ##############
ROUGE_PATH = "~/ROUGE/"
```



## Run Cmdline

LSTM + Sequence Labeling:

```shell
python train.py --cuda --gpu <gpuid> --sentence_encoder deeplstm --sentence_decoder SeqLab --save_root <savedir> --log_root <logdir> --lr_descent --grad_clip --max_grad_norm 10 --use_pyrouge
```

Transformer + Sequence Labeling:

```shell
python train.py --cuda --gpu <gpuid> --sentence_encoder transformer --sentence_decoder SeqLab --save_root <savedir> --log_root <logdir> --lr_descent --grad_clip --max_grad_norm 10 --use_pyrouge
```

