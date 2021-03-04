from fastNLP.io.file_utils import get_cache_path
from summarizationLoader import ArxivLoader

if __name__ == "__main__":

    # 请设置fastNLP默认cache的存放路径FASTNLP_CACHE_DIR, get_cache_path会获取设置下载的数据位置
    # 详细可参考: https://gitee.com/fastnlp/fastNLP/blob/7b4e099c5267efb6a4a88b9d789a0940be05bb56/fastNLP/io/file_utils.py#L228
    print(f"下载的数据位置: {get_cache_path()}")
    ArxivLoader().download()
    data = ArxivLoader().load()
    print(data)