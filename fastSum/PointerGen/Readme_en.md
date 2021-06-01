# Pointer Generator

Code for paper *[Get To The Point Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)* implemented by FastNLP.
Source code: https://github.com/abisee/pointer-generator



## Input data
The type of input data needs to be jsonl. The input file contains two keys:

- text:  original text
- summary:  abstract of the text

The type of value needs to be list.

E.g.

- "text": ["london -lrb- cnn -rrb- a 19-year-old man was charged wednesday with terror offenses after he was arrested as he returned to britain from turkey , london 's metropolitan police said .", "yahya rashid , a uk national from northwest london , was detained at luton airport on tuesday after he arrived on a flight from istanbul , police said .", "he 's been charged with engaging in conduct in preparation of acts of terrorism , and with engaging in conduct with the intention of assisting others to commit acts of terrorism . both charges relate to the period between november 1 and march 31 .", "rashid is due to appear in westminster magistrates ' court on wednesday , police said .", "cnn 's lindsay isaac contributed to this report ."]
- "summary": ["london 's metropolitan police say the man was arrested at luton airport after landing on a flight from istanbul .", "he 's been charged with terror offenses allegedly committed since the start of november ."]



## Run Cmdline

### Train
Command line for training:
```shell
python train.py -train_data_path TRAIN_DATA_PATH -eval_data_path VALID_DATA_PATH -log_root LOG_ROOT_NAME -is_pointer_gen -is_coverage -n_epochs 33 -visible_gpu 0 -lr_coverage 0.025 -batch_size 16
```
- is_pointer_gen: use pointer
- is_coverage: use coverage



### Test

Command line for testing:

```shell
python decode.py -decode_data_path TEST_DATA_PATH -train_data_path TRAIN_DATA_PATH -test_model CHECKPOINT -log_root LOG_ROOT_NAME -is_pointer_gen -is_coverage -test_data_name TEST_DATA_NAME -visible_gpu 0
```
- LOG_ROOT_NAME: root to save result
- is_pointer_gen and is_coverage need to be the same as training.
