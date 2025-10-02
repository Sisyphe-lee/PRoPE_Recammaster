#HF_ENDPOINT = "https://hf-mirror.com"
from datasets import load_dataset

HF_MIRROR_URL = "https://hf-mirror.com"
DATASET_PATH = "/nas/datasets//vipe_wild_10k"
DATASET_NAME = "nvidia/vipe-wild-sdg-1m"

# 获取前 10000 个样本
# 如果数据集有多个分割（如 train, test, validation）
dataset = load_dataset(DATASET_NAME, split="all")
subset = dataset["all"].select(range(1000))

# 如果数据集只有一个分割
subset = dataset.select(range(10000))

# 保存到本地
subset.save_to_disk(DATASET_PATH)