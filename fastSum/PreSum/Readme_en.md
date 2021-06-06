# PreSum
Code for EMNLP 2019 paper *[Text Summarization with Pretrained Encoders](https://arxiv.org/pdf/1908.08345)* implemented by FastNLP.

Source code: https://github.com/nlpyang/PreSumm



## Data preprocessing
Use preprocess.py to preprocess the data. Examples are as follows: 
```shell
python preprocess.py --raw_path INPUT_PATH --save_path data/OUTPUT_PATH --log_file LOG_PATH
```
`INPUT_PATH`,` OUTPUT_PATH` and `LOG_PATH` represent the preprocessing input directory, output directory and log path respectively. 

There are files named `xx.train.jsonl`, `xx.val.jsonl`, `xx.test.jsonl` in the `INPUT_PATH`. These files need to contain two keys: 

- text:  original text
- summary:  abstract of the text

The type of value needs to be list.



## Run Cmdline

### TransformerABS

**Train transformer based model for abstractive summarization:**

```shell
python train_presum.py -task abs -mode train -dec_dropout 0.2 -save_path SAVE_DIR -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 16 -accum_count 5 -use_bert_emb true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 0,1,2,3 -log_file LOG_PATH -label_type INPUT_DIR -valid_steps 2000 -n_epochs 10 -max_summary_len 600
```
- SAVE_DIR: directory to save the training model
- LOG_PATH: path of the log
- INPUT_DIR: directory of the input data, which corresponds to the OUTPUT_PATH in Data preprocessing part (You just need to tell the folder name you named in Data preprocessing.)

**Command line for testing:**

```shell
python train_presum.py -task abs -mode test -test_batch_size 12 -log_file LOG_PATH -test_from CHECKPOINT_PATH -sep_optim true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -block_trigram True -label_type INPUT_DIR -max_summary_len 600 -decode_path DECODE_PATH
```
- LOG_PATH: path of the log
- CHECKPOINT_PATH: path of the checkpoint
- INPUT_DIR: directory of the test set, which corresponds to the OUTPUT_PATH in Data preprocessing part
- DECODE_PATH: path to save gold summary and generated summary
- max_ length, min_ Length and alpha need to be adjusted according to the characteristics of different datasets



### BERTSUMABS
**Train BERT based model for abstractive summarization:**

```shell
python train_presum.py -task abs -mode train -dec_dropout 0.1 -save_path SAVE_DIR  -sep_optim False -lr 0.05 -save_checkpoint_steps 2000 -batch_size 8 -accum_count 8 -use_bert_emb true -warmup_steps 10000 -max_pos 512 -visible_gpus 0,1,2,3 -log_file LOG_PATH -label_type INPUT_DIR -valid_steps 2000 -n_epochs 10 -max_summary_len 600 -encoder baseline -enc_dropout 0.1 -enc_hidden_size 512  -enc_layers 6 -enc_ff_size 2048 -dec_layers 6 -dec_hidden_size 512 -dec_ff_size 2048
```
- SAVE_DIR: directory to save the training model
- LOG_PATH: path of the log
- INPUT_DIR: directory of the input data, which corresponds to the OUTPUT_PATH in Data preprocessing part (You just need to tell the folder name you named in Data preprocessing.)

**Command line for testing:**

```shell
python train_presum.py -task abs -mode test -test_batch_size 12 -log_file LOG_PATH -test_from CHECKPOINT_PATH -sep_optim False -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -block_trigram True -label_type INPUT_DIR -max_summary_len 300 -decode_path DECODE_PATH -encoder baseline
```
- LOG_PATH: path of the log
- CHECKPOINT_PATH: path of the checkpoint
- INPUT_DIR: directory of the test set, which corresponds to the OUTPUT_PATH in Data preprocessing part
- DECODE_PATH: path to save gold summary and generated summary
- max_ length, min_ Length and alpha need to be adjusted according to the characteristics of different datasets



### TransformerEXT
**Train transformer based model for extractive summarization:**

```shell
python train_presum.py -task ext -mode train -ext_dropout 0.1 -save_path SAVE_DIR -lr 2e-3 -save_checkpoint_steps 2000 -batch_size 16 -accum_count 4 -warmup_steps 10000 -max_pos 512 -visible_gpus 0,1,2,3 -log_file LOG_PATH -label_type INPUT_DIR -valid_steps 2000 -n_epochs 10 -max_summary_len 600 -encoder baseline -ext_hidden_size 512 -ext_layers 6 -ext_ff_size 2048
```
- SAVE_DIR: directory to save the training model
- LOG_PATH: path of the log
- INPUT_DIR: directory of the input data, which corresponds to the OUTPUT_PATH in Data preprocessing part (You just need to tell the folder name you named in Data preprocessing.)

**Command line for testing:**

```shell
python train_presum.py -task ext -mode test -test_batch_size 12 -log_file LOG_PATH -test_from CHECKPOINT_PATH -sep_optim False -visible_gpus 1 -max_pos 512 -block_trigram True -label_type INPUT_DIR -max_summary_len 600 -decode_path DECODE_PATH -encoder baseline -ext_hidden_size 512 -ext_layers 6 -ext_ff_size 2048

```
- LOG_PATH: path of the log
- CHECKPOINT_PATH: path of the checkpoint
- INPUT_DIR: directory of the test set, which corresponds to the OUTPUT_PATH in Data preprocessing part
- DECODE_PATH: path to save gold summary and generated summary



### BERTSUMEXT
**Train BERT based model for extractive summarization:**

```shell
python train_presum.py -task ext -mode train -ext_dropout 0.1 -save_path SAVE_DIR -lr 2e-3 -save_checkpoint_steps 2000 -batch_size 16 -accum_count 4 -warmup_steps 10000 -max_pos 512 -visible_gpus 4,5,6,7 -log_file LOG_PATH -label_type INPUT_DIR -valid_steps 2000 -n_epochs 10 -max_summary_len 600 -sep_optim False
```
- SAVE_DIR: directory to save the training model
- LOG_PATH: path of the log
- INPUT_DIR: directory of the input data, which corresponds to the OUTPUT_PATH in Data preprocessing part (You just need to tell the folder name you named in Data preprocessing.)

**Command line for testing:**

```shell
python train_presum.py -task ext -mode test -test_batch_size 12 -log_file LOG_PATH -test_from CHECKPOINT_PATH -sep_optim False -visible_gpus 0 -max_pos 512 -block_trigram True -label_type INPUT_DIR -max_summary_len 600 -decode_path DECODE_PATH

```
- LOG_PATH: path of the log
- CHECKPOINT_PATH: path of the checkpoint
- INPUT_DIR: directory of the test set, which corresponds to the OUTPUT_PATH in Data preprocessing part
- DECODE_PATH: path to save gold summary and generated summary
