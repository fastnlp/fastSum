from typing import Dict, Optional
import os
import random
from pathlib import Path

from fastNLP.io.loader import JsonLoader
from fastNLP.io.data_bundle import DataBundle
from fastNLP.core.const import Const
from fastNLP.io.file_utils import get_cache_path, _get_dataset_url, cached_path

DATASET_DIR = {
    # Summarization
    "ami": "AMI.zip",
    "arxiv": "Arxiv.zip",
    "billsum": "BillSum.zip",
    "cnndm": "CNNDM.zip",
    "icsi": "ICSI.zip",
    "multi-news": "Multi-News.zip",
    "pubmed": "Pubmed.zip",
    "reddit tifu": "Reddit TIFU.zip",
    "samsum": "SAMSum.zip",
    "wikihow": "WikiHow.zip",
    "xsum": "Xsum.zip",
}


class SumLoader(JsonLoader):
    """
    所有摘要数据集loader的父类
    """

    DATASET_NAME = None  # 对应 DATASET_DIR 中的 key

    def __init__(self, fields: Optional[Dict[str, str]] = None):
        if fields is None:
            fields = {"text": "text", "summary": "summary", "label": Const.TARGET}
        super(SumLoader, self).__init__(fields=fields)

    def load(self, paths: Optional[Path] = None) -> DataBundle:
        pass

    def download(self):
        default_cache_path = get_cache_path()
        url = _get_dataset_url(self.DATASET_NAME, DATASET_DIR)
        output_dir = cached_path(
            url_or_filename=url, cache_dir=default_cache_path, name="dataset"
        )
        # https://gitee.com/fastnlp/fastNLP/blob/7b4e099c5267efb6a4a88b9d789a0940be05bb56/fastNLP/io/file_utils.py#L201
        # 如果只有一个文件, get_filepath 返回 filepath + filename
        # os.path.dirname 反向处理
        if os.path.isfile(output_dir):
            output_dir = os.path.dirname(output_dir)
        return output_dir


class CNNDMLoader(SumLoader):
    """
    CNNDM数据集的loader
    如果您的文章使用了这份数据，请引用

    https://www.aclweb.org/anthology/K16-1028/
    """

    DATASET_NAME = "cnndm"

    def __init__(self):
        super(CNNDMLoader, self).__init__()

    def load(self, paths: Optional[Path] = None) -> DataBundle:
        if paths is None:
            paths = self.download()

        _paths = {}
        if paths:
            if os.path.isdir(paths):
                if not os.path.isfile(os.path.join(paths, "CNNDM.train.label.jsonl")):
                    raise FileNotFoundError(
                        f"CNNDM.train.label.jsonl is not found in {paths}"
                    )
                _paths["train"] = os.path.join(paths, "CNNDM.train.label.jsonl")
                _paths["dev"] = os.path.join(paths, "CNNDM.valid.label.jsonl")
                _paths["test"] = os.path.join(paths, "CNNDM.test.label.jsonl")
                paths = _paths
            else:
                raise NotADirectoryError(f"{paths} is not a valid directory.")

        datasets = {name: self._load(path) for name, path in paths.items()}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class ArxivLoader(SumLoader):
    """
    Arxiv数据集的loader
    如果您的文章使用了这份数据，请引用

    https://arxiv.org/abs/1804.05685
    """

    DATASET_NAME = "arxiv"

    def __init__(self):
        super(ArxivLoader, self).__init__()

    def load(self, paths: Optional[Path] = None) -> DataBundle:
        if paths is None:
            paths = self.download()

        _paths = {}
        if paths:
            if os.path.isdir(paths):
                if not os.path.isfile(os.path.join(paths, "arxiv.train.label.jsonl")):
                    raise FileNotFoundError(
                        f"arxiv.train.label.jsonl is not found in {paths}"
                    )
                _paths["train"] = os.path.join(paths, "arxiv.train.label.jsonl")
                _paths["dev"] = os.path.join(paths, "arxiv.valid.label.jsonl")
                _paths["test"] = os.path.join(paths, "arxiv.test.label.jsonl")
                paths = _paths
            else:
                raise NotADirectoryError(f"{paths} is not a valid directory.")

        datasets = {name: self._load(path) for name, path in paths.items()}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class BillSumLoader(SumLoader):
    """
    BillSum数据集的loader
    如果您的文章使用了这份数据，请引用

    https://arxiv.org/abs/1910.00523
    """

    DATASET_NAME = "billsum"

    def __init__(self):
        super(BillSumLoader, self).__init__()

    def load(self, paths: Optional[Path] = None) -> DataBundle:
        if paths is None:
            paths = self.download()

        _paths = {}
        if paths:
            if os.path.isdir(paths):
                if not os.path.isfile(
                    os.path.join(paths, "billsum_us.train.label.jsonl")
                ):
                    raise FileNotFoundError(
                        f"billsum_us.train.label.jsonl is not found in {paths}"
                    )
                _paths["train"] = os.path.join(paths, "billsum_us.train.label.jsonl")
                _paths["dev"] = os.path.join(paths, "billsum_ca.valid.label.jsonl")
                _paths["test"] = os.path.join(paths, "billsum_us.test.label.jsonl")
                paths = _paths
            else:
                raise NotADirectoryError(f"{paths} is not a valid directory.")

        datasets = {name: self._load(path) for name, path in paths.items()}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class MultiNewsLoader(SumLoader):
    """
    MultiNews数据集的loader
    如果您的文章使用了这份数据，请引用

    https://arxiv.org/abs/1906.01749
    """

    DATASET_NAME = "multi-news"

    def __init__(self):
        super(MultiNewsLoader, self).__init__()

    def load(self, paths: Optional[Path] = None) -> DataBundle:
        if paths is None:
            paths = self.download()

        _paths = {}
        if paths:
            if os.path.isdir(paths):
                if not os.path.isfile(
                    os.path.join(paths, "multinews.train.label.jsonl")
                ):
                    raise FileNotFoundError(
                        f"multinews.train.label.jsonl is not found in {paths}"
                    )
                _paths["train"] = os.path.join(paths, "multinews.train.label.jsonl")
                _paths["dev"] = os.path.join(paths, "multinews.valid.label.jsonl")
                _paths["test"] = os.path.join(paths, "multinews.test.label.jsonl")
                paths = _paths
            else:
                raise NotADirectoryError(f"{paths} is not a valid directory.")

        datasets = {name: self._load(path) for name, path in paths.items()}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class PubmedLoader(SumLoader):
    """
    Pubmed数据集的loader
    如果您的文章使用了这份数据，请引用

    https://arxiv.org/abs/1804.05685
    """

    DATASET_NAME = "pubmed"

    def __init__(self):
        super(PubmedLoader, self).__init__()

    def load(self, paths: Optional[Path] = None) -> DataBundle:
        if paths is None:
            paths = self.download()

        _paths = {}
        if paths:
            if os.path.isdir(paths):
                if not os.path.isfile(os.path.join(paths, "pubmed.train.label.jsonl")):
                    raise FileNotFoundError(
                        f"pubmed.train.label.jsonl is not found in {paths}"
                    )
                _paths["train"] = os.path.join(paths, "pubmed.train.label.jsonl")
                _paths["dev"] = os.path.join(paths, "pubmed.valid.label.jsonl")
                _paths["test"] = os.path.join(paths, "pubmed.test.label.jsonl")
                paths = _paths
            else:
                raise NotADirectoryError(f"{paths} is not a valid directory.")

        datasets = {name: self._load(path) for name, path in paths.items()}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class SAMSumLoader(SumLoader):
    """
    SAMSum数据集的loader
    如果您的文章使用了这份数据，请引用

    https://arxiv.org/abs/1911.12237
    """

    DATASET_NAME = "samsum"

    def __init__(self):
        super(SAMSumLoader, self).__init__()

    def load(self, paths: Optional[Path] = None) -> DataBundle:
        if paths is None:
            paths = self.download()

        _paths = {}
        if paths:
            if os.path.isdir(paths):
                if not os.path.isfile(os.path.join(paths, "SAMSum.train.label.jsonl")):
                    raise FileNotFoundError(
                        f"SAMSum.train.label.jsonl is not found in {paths}"
                    )
                _paths["train"] = os.path.join(paths, "SAMSum.train.label.jsonl")
                _paths["dev"] = os.path.join(paths, "SAMSum.valid.label.jsonl")
                _paths["test"] = os.path.join(paths, "SAMSum.test.label.jsonl")
                paths = _paths
            else:
                raise NotADirectoryError(f"{paths} is not a valid directory.")

        datasets = {name: self._load(path) for name, path in paths.items()}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class WikiHowLoader(SumLoader):
    """
    WikiHow数据集的loader
    如果您的文章使用了这份数据，请引用

    https://arxiv.org/abs/1810.09305
    """

    DATASET_NAME = "wikihow"

    def __init__(self):
        super(WikiHowLoader, self).__init__()

    def load(self, paths: Optional[Path] = None) -> DataBundle:
        if paths is None:
            paths = self.download()

        _paths = {}
        if paths:
            if os.path.isdir(paths):
                if not os.path.isfile(os.path.join(paths, "wikihow.train.label.jsonl")):
                    raise FileNotFoundError(
                        f"wikihow.train.label.jsonl is not found in {paths}"
                    )
                _paths["train"] = os.path.join(paths, "wikihow.train.label.jsonl")
                _paths["dev"] = os.path.join(paths, "wikihow.val.label.jsonl")
                _paths["test"] = os.path.join(paths, "wikihow.test.label.jsonl")
                paths = _paths
            else:
                raise NotADirectoryError(f"{paths} is not a valid directory.")

        datasets = {name: self._load(path) for name, path in paths.items()}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class XsumLoader(SumLoader):
    """
    Xsum数据集的loader
    如果您的文章使用了这份数据，请引用

    https://arxiv.org/abs/1808.08745
    """

    DATASET_NAME = "xsum"

    def __init__(self):
        super(XsumLoader, self).__init__()

    def load(self, paths: Optional[Path] = None) -> DataBundle:
        if paths is None:
            paths = self.download()

        _paths = {}
        if paths:
            if os.path.isdir(paths):
                if not os.path.isfile(os.path.join(paths, "xsum.train.label.jsonl")):
                    raise FileNotFoundError(
                        f"xsum.train.label.jsonl is not found in {paths}"
                    )
                _paths["train"] = os.path.join(paths, "xsum.train.label.jsonl")
                _paths["dev"] = os.path.join(paths, "xsum.valid.label.jsonl")
                _paths["test"] = os.path.join(paths, "xsum.test.label.jsonl")
                paths = _paths
            else:
                raise NotADirectoryError(f"{paths} is not a valid directory.")

        datasets = {name: self._load(path) for name, path in paths.items()}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class RedditTIFULoader(SumLoader):
    """
    Reddit TIFU数据集的loader
    如果您的文章使用了这份数据，请引用

    https://arxiv.org/abs/1811.00783
    """

    DATASET_NAME = "reddit tifu"

    def __init__(self, tag, valid_ratio=0.05, test_ratio=0.05):
        super(RedditTIFULoader, self).__init__()
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        assert tag in ["long", "short"], "tag not valid (neither long nor short)!"
        self.tag = tag

    def load(self, paths: Optional[Path] = None) -> DataBundle:
        if paths is None:
            paths = self.download()

        _paths = {}
        if paths:
            if os.path.isdir(paths):
                if not os.path.isfile(
                    os.path.join(paths, f"tifu_{self.tag}.all.label.jsonl")
                ):
                    raise FileNotFoundError(
                        f"tifu_{self.tag}.all.label.jsonl is not found in {paths}"
                    )

                _split_set(
                    f"tifu_{self.tag}.all.label",
                    paths,
                    split_name1="middev",
                    split_name2="train",
                    ratio=self.valid_ratio + self.test_ratio,
                )
                if self.valid_ratio + self.test_ratio > 0:
                    _split_set(
                        "middev",
                        paths,
                        split_name1="test",
                        split_name2="dev",
                        ratio=self.test_ratio / (self.valid_ratio + self.test_ratio),
                    )
                _paths["train"] = os.path.join(paths, "train.jsonl")
                if self.valid_ratio > 0:
                    _paths["dev"] = os.path.join(paths, "dev.jsonl")
                if self.test_ratio > 0:
                    _paths["test"] = os.path.join(paths, "test.jsonl")
                paths = _paths
            else:
                raise NotADirectoryError(f"{paths} is not a valid directory.")

        datasets = {name: self._load(path) for name, path in paths.items()}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class AMILoader(SumLoader):
    """
    AMI数据集的loader
    如果您的文章使用了这份数据，请引用

    http://groups.inf.ed.ac.uk/ami/download/
    """

    DATASET_NAME = "ami"

    def __init__(self, valid_ratio=0.05, test_ratio=0.05):
        # AMI 没有 label
        fields = {
            "text": "text",
            "summary": "summary",
        }
        super(AMILoader, self).__init__(fields)

        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio

    def load(self, paths: Optional[Path] = None) -> DataBundle:
        if paths is None:
            paths = self.download()

        _paths = {}
        if paths:
            if os.path.isdir(paths):
                if not os.path.isfile(os.path.join(paths, "AMI.jsonl")):
                    raise FileNotFoundError(f"AMI.jsonl is not found in {paths}")

                _split_set(
                    "AMI",
                    paths,
                    split_name1="middev",
                    split_name2="train",
                    ratio=self.valid_ratio + self.test_ratio,
                )
                if self.valid_ratio + self.test_ratio > 0:
                    _split_set(
                        "middev",
                        paths,
                        split_name1="test",
                        split_name2="dev",
                        ratio=self.test_ratio / (self.valid_ratio + self.test_ratio),
                    )
                _paths["train"] = os.path.join(paths, "train.jsonl")
                if self.valid_ratio > 0:
                    _paths["dev"] = os.path.join(paths, "dev.jsonl")
                if self.test_ratio > 0:
                    _paths["test"] = os.path.join(paths, "test.jsonl")
                paths = _paths
            else:
                raise NotADirectoryError(f"{paths} is not a valid directory.")

        datasets = {name: self._load(path) for name, path in paths.items()}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class ICSILoader(SumLoader):
    """
    ICSI数据集的loader
    如果您的文章使用了这份数据，请引用

    http://groups.inf.ed.ac.uk/ami/icsi/
    """

    DATASET_NAME = "icsi"

    def __init__(self, valid_ratio=0.05, test_ratio=0.05):
        # ICSI 没有 label
        fields = {
            "text": "text",
            "summary": "summary",
        }
        super(ICSILoader, self).__init__(fields)
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio

    def load(self, paths: Optional[Path] = None) -> DataBundle:
        if paths is None:
            paths = self.download()

        _paths = {}
        if paths:
            if os.path.isdir(paths):
                if not os.path.isfile(os.path.join(paths, "ICSI.jsonl")):
                    raise FileNotFoundError(f"ICSI.jsonl is not found in {paths}")

                _split_set(
                    "ICSI",
                    paths,
                    split_name1="middev",
                    split_name2="train",
                    ratio=self.valid_ratio + self.test_ratio,
                )
                if self.valid_ratio + self.test_ratio > 0:
                    _split_set(
                        "middev",
                        paths,
                        split_name1="test",
                        split_name2="dev",
                        ratio=self.test_ratio / (self.valid_ratio + self.test_ratio),
                    )
                _paths["train"] = os.path.join(paths, "train.jsonl")
                if self.valid_ratio > 0:
                    _paths["dev"] = os.path.join(paths, "dev.jsonl")
                if self.test_ratio > 0:
                    _paths["test"] = os.path.join(paths, "test.jsonl")
                paths = _paths
            else:
                raise NotADirectoryError(f"{paths} is not a valid directory.")

        datasets = {name: self._load(path) for name, path in paths.items()}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


def _split_set(
    dataset_name,
    data_dir,
    split_name1="dev",
    split_name2="train",
    ratio=0.0,
    suffix="jsonl",
    keep_orig: bool = True,
):
    if ratio == 0:
        os.renames(
            os.path.join(data_dir, f"{dataset_name}.{suffix}"),
            os.path.join(data_dir, f"{split_name2}.{suffix}"),
        )
        return data_dir

    if not os.path.exists(os.path.join(data_dir, f"{split_name1}.{suffix}")):
        if ratio > 0:
            assert 0 < ratio < 1, "dev_ratio should be in range (0,1)."
            try:
                with open(
                    os.path.join(data_dir, f"{dataset_name}.{suffix}"),
                    "r",
                    encoding="utf-8",
                ) as f, open(
                    os.path.join(data_dir, f"middle_file.{suffix}"),
                    "w",
                    encoding="utf-8",
                ) as f1, open(
                    os.path.join(data_dir, f"{split_name1}.{suffix}"),
                    "w",
                    encoding="utf-8",
                ) as f2:
                    for line in f:
                        if random.random() < ratio:
                            f2.write(line)
                        else:
                            f1.write(line)
                if keep_orig:
                    assert split_name1 != dataset_name and split_name2 != dataset_name
                else:
                    os.remove(os.path.join(data_dir, f"{dataset_name}.{suffix}"))
                os.renames(
                    os.path.join(data_dir, f"middle_file.{suffix}"),
                    os.path.join(data_dir, f"{split_name2}.{suffix}"),
                )
            finally:
                if os.path.exists(os.path.join(data_dir, f"middle_file.{suffix}")):
                    os.remove(os.path.join(data_dir, f"middle_file.{suffix}"))

    return data_dir