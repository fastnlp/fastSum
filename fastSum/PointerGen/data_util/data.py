from fastNLP import DataSet
from fastNLP.io.base_loader import DataBundle
from fastNLP.io import DataSetLoader, JsonLoader
from fastNLP import Vocabulary
from nltk.tokenize import sent_tokenize

import glob
import numpy as np
import struct
from tensorflow.core.example import example_pb2
from data_util.logging import logger

# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]'  # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]'  # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]'  # This has a vocab id, which is used at the end of untruncated target sequences


# Note: none of <s>, </s>, [PAD], [UNK], [START], [STOP] should appear in the vocab file.


def convert_list_to_ndarray(field):
    if isinstance(field, list):
        return np.array(field)
    return field


def prepare_dataInfo(mode, vocab_size, config, train_data_path=None, dev_data_path=None, test_data_path=None):
    def sent_to_words(sents):
        result = []
        for sent in sents:
            result.extend([word.strip() for word in sent.split(" ") if len(word.strip()) != 0])
        return result

    # dataloader = Cnn_dailymailLodaer()
    # 适用于输入是json的文件，每个json必须有field ：text和summary，二者都是tokenized
    dataloader = JsonLoader(fields={"text": "words", "summary": "abstract_sentences"})
    if mode == 'train':
        if train_data_path is None or dev_data_path is None:
            print("training with no train data path or dev data path! ")
        paths = {"train": train_data_path, "dev": dev_data_path}
    else:
        if test_data_path is None:
            print("testing with no test data path ! ")
        paths = {"train": train_data_path, "test": test_data_path}
    # dataInfo = dataloader.process(paths, vocab_path, vocab_size)
    print("=" * 10)
    print(paths)
    dataInfo = dataloader.load(paths)
    for key, _dataset in dataInfo.datasets.items():
        _dataset.apply(lambda ins: " ".join(ins['words']), new_field_name='article')
        _dataset.apply(lambda ins: sent_to_words(ins['words']), new_field_name='words')
        _dataset.apply(lambda ins: sent_tokenize(" ".join(ins['abstract_sentences'])),
                       new_field_name='abstract_sentences')

    vocab = Vocabulary(max_size=vocab_size - 2, padding=PAD_TOKEN, unknown=UNKNOWN_TOKEN)
    vocab.from_dataset(dataInfo.datasets['train'], field_name='words')
    vocab.add(START_DECODING)
    vocab.add(STOP_DECODING)
    print(vocab.to_word(0))
    print(len(vocab))
    assert vocab_size == len(vocab), "vocab_size error!!!"
    dataInfo.set_vocab(vocab, "train")

    for key, dataset in dataInfo.datasets.items():
        data_dict = {"enc_len": [],
                     "enc_input": [],
                     "dec_input": [],
                     "target": [],
                     "dec_len": [],
                     "article_oovs": [],
                     "enc_input_extend_vocab": []}

        for instance in dataset:
            article = instance["article"]
            abstract_sentences = instance["abstract_sentences"]

            enc_len, enc_input, dec_input, target, dec_len, article_oovs, enc_input_extend_vocab = getting_full_info(
                article, abstract_sentences, dataInfo.vocabs['train'], config)

            data_dict["enc_len"].append(enc_len)
            data_dict["enc_input"].append(enc_input)
            data_dict["dec_input"].append(dec_input)
            data_dict["target"].append(target)
            data_dict["dec_len"].append(dec_len)
            data_dict["article_oovs"].append(article_oovs)
            data_dict["enc_input_extend_vocab"].append(enc_input_extend_vocab)

        logger.info("-----prepare_dataInfo for dataset " + key + "-----")
        logger.info(str(len(data_dict["enc_len"])) + " " + str(len(data_dict["enc_input"])) + " " + str(
            len(data_dict["dec_input"])) + " " +
                    str(len(data_dict["target"])) + " " + str(len(data_dict["dec_len"])) + " " + str(
            len(data_dict["article_oovs"])) + " " +
                    str(len(data_dict["enc_input_extend_vocab"])))
        dataset.add_field("enc_len", data_dict["enc_len"])
        dataset.add_field("enc_input", data_dict["enc_input"])
        dataset.add_field("dec_input", data_dict["dec_input"])
        dataset.add_field("target", data_dict["target"])
        dataset.add_field("dec_len", data_dict["dec_len"])
        dataset.add_field("article_oovs", data_dict["article_oovs"])
        dataset.add_field("enc_input_extend_vocab", data_dict["enc_input_extend_vocab"])

        dataset.set_input("enc_len", "enc_input", "dec_input", "dec_len", "article_oovs", "enc_input_extend_vocab")
        dataset.set_target("target", "article_oovs", "abstract_sentences")
    '''
    for name, dataset in dataInfo.datasets.items():
        for field_name in dataset.get_field_names():
            dataset.apply_field(convert_list_to_ndarray, field_name=field_name, new_field_name=field_name)
    '''
    return dataInfo


def get_dec_inp_targ_seqs(sequence, max_len, start_id, stop_id):
    inp = [start_id] + sequence[:]
    target = sequence[:]
    if len(inp) > max_len:  # truncate
        inp = inp[:max_len]
        target = target[:max_len]  # no end_token
    else:  # no truncation
        target.append(stop_id)  # end token
    assert len(inp) == len(target)
    return inp, target


def getting_full_info(article, abstract_sentences, vocab, config):
    # Get ids of special tokens
    start_decoding = vocab.to_index(START_DECODING)
    stop_decoding = vocab.to_index(STOP_DECODING)

    # Process the article
    article_words = article.split()
    if len(article_words) > config.max_enc_steps:
        article_words = article_words[:config.max_enc_steps]
    enc_len = len(article_words)  # store the length after truncation but before padding
    enc_input = [vocab.to_index(w) for w in
                 article_words]  # list of word ids; OOVs are represented by the id for UNK token

    # Process the abstract
    abstract = ' '.join(abstract_sentences)  # string
    abstract_words = abstract.split()  # list of strings
    abs_ids = [vocab.to_index(w) for w in
               abstract_words]  # list of word ids; OOVs are represented by the id for UNK token

    # Get the decoder input sequence and target sequence
    dec_input, target = get_dec_inp_targ_seqs(abs_ids, config.max_dec_steps, start_decoding, stop_decoding)
    dec_len = len(dec_input)

    # If using pointer-generator mode, we need to store some extra info
    if config.pointer_gen:
        # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id; also store the in-article OOVs words themselves
        enc_input_extend_vocab, article_oovs = article2ids(article_words, vocab)

        # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
        abs_ids_extend_vocab = abstract2ids(abstract_words, vocab, article_oovs)

        # Overwrite decoder target sequence so it uses the temp article OOV ids
        _, target = get_dec_inp_targ_seqs(abs_ids_extend_vocab, config.max_dec_steps, start_decoding, stop_decoding)
    else:
        article_oovs = ["N O N E"]
        enc_input_extend_vocab = [-1]

    return enc_len, enc_input, dec_input, target, dec_len, article_oovs, enc_input_extend_vocab


def article2ids(article_words, vocab):
    ids = []
    oovs = []
    unk_id = vocab.to_index(UNKNOWN_TOKEN)
    for w in article_words:
        i = vocab.to_index(w)
        if i == unk_id:  # If w is OOV
            if w not in oovs:  # Add to list of OOVs
                oovs.append(w)
            oov_num = oovs.index(w)  # This is 0 for the first article OOV, 1 for the second article OOV...
            ids.append(len(vocab) + oov_num)  # This is e.g. 50000 for the first article OOV, 50001 for the second...
        else:
            ids.append(i)
    if len(oovs) == 0:
        oovs.append("N O N E")
    return ids, oovs


def abstract2ids(abstract_words, vocab, article_oovs):
    ids = []
    unk_id = vocab.to_index(UNKNOWN_TOKEN)
    for w in abstract_words:
        i = vocab.to_index(w)
        if i == unk_id:  # If w is an OOV word
            if w in article_oovs:  # If w is an in-article OOV
                vocab_idx = len(vocab) + article_oovs.index(w)  # Map to its temporary article OOV number
                ids.append(vocab_idx)
            else:  # If w is an out-of-article OOV
                ids.append(unk_id)  # Map to the UNK token id
        else:
            ids.append(i)
    return ids


def outputids2words(id_list, vocab, article_oovs):
    words = []
    for i in id_list:
        try:
            w = vocab.to_word(i)  # might be [UNK]
        except KeyError as e:  # w is OOV
            # assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            assert "N O N E" not in article_oovs, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            article_oov_idx = i - len(vocab)
            try:
                w = article_oovs[article_oov_idx]
            except ValueError as e:  # i doesn't correspond to an article oov
                raise ValueError(
                    'Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (
                        i, article_oov_idx, len(article_oovs)))
        words.append(w)
    return words


def abstract2sents(abstract):
    cur = 0
    sents = []
    while True:
        try:
            start_p = abstract.index(SENTENCE_START, cur)
            end_p = abstract.index(SENTENCE_END, start_p + 1)
            cur = end_p + len(SENTENCE_END)
            sents.append(abstract[start_p + len(SENTENCE_START):end_p])
        except ValueError as e:  # no more sentences
            return sents


def show_art_oovs(article, vocab):
    unk_token = vocab.to_index(UNKNOWN_TOKEN)
    words = article.split(' ')
    words = [("__%s__" % w) if vocab.to_index(w) == unk_token else w for w in words]
    out_str = ' '.join(words)
    return out_str


def show_abs_oovs(abstract, vocab, article_oovs):
    unk_token = vocab.to_index(UNKNOWN_TOKEN)
    words = abstract.split(' ')
    new_words = []
    for w in words:
        # w is oov
        if vocab.to_index(w) == unk_token:
            # if article_oovs is None:  # baseline mode
            if "N O N E" in article_oovs:
                new_words.append("__%s__" % w)
            else:  # pointer-generator mode
                if w in article_oovs:
                    new_words.append("__%s__" % w)
                else:
                    new_words.append("!!__%s__!!" % w)
        else:  # w is in-vocab word
            new_words.append(w)

    out_str = ' '.join(new_words)
    return out_str
