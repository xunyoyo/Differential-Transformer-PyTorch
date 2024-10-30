# data/dataset.py
# import upper route
import sys 
sys.path.append("/Volumes/xunyoyo/ACL/Differential-Transformer-PyTorch") 

from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from datas.tokenizer import Tokenizer

class HuggingFaceDataset(Dataset):
    """
    从 Hugging Face 数据集中加载数据并进行分词的自定义数据集类。
    """
    def __init__(self, dataset_name="wikitext", config_name="wikitext-103-raw-v1", split="train", tokenizer=None, max_length=512):
        """
        初始化数据集，加载数据并进行分词。

        Args:
            dataset_name (str): Hugging Face 数据集名称。
            config_name (str): 数据集的配置名称。
            split (str): 数据集的部分 ('train', 'test', 'validation')。
            tokenizer (Tokenizer): 分词器实例。
            max_length (int): 文本的最大长度（超过此长度将被截断）。
        """
        # 使用 Hugging Face datasets 库加载数据集
        self.dataset = load_dataset(dataset_name, config_name, split=split)
        self.tokenizer = tokenizer if tokenizer else Tokenizer()
        self.max_length = max_length
        self.data = self._tokenize_data()

    def _tokenize_data(self):
        """
        对数据集中的所有文本进行分词，并截断到最大长度。

        Returns:
            list[list[int]]: 包含所有文本的 token ID 列表。
        """
        tokenized_texts = []
        for item in self.dataset:
            text = item['text']
            token_ids = self.tokenizer.encode(text)
            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]  # 截断文本
            tokenized_texts.append(token_ids)
        return tokenized_texts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取指定索引的数据项，自动将 token ID 转换为 PyTorch 的 Tensor。

        Args:
            idx (int): 数据项的索引。

        Returns:
            dict: 包含输入 ID 和注意力掩码的字典。
        """
        token_ids = self.data[idx]
        attention_mask = [1] * len(token_ids)

        # 将 token_ids 填充到 max_length
        padding_length = self.max_length - len(token_ids)
        if padding_length > 0:
            token_ids = token_ids + [0] * padding_length
            attention_mask = attention_mask + [0] * padding_length

        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }

# 示例使用
if __name__ == "__main__":
    print(sys.path)
    # 初始化分词器和数据集
    tokenizer = Tokenizer()
    dataset = HuggingFaceDataset(dataset_name="wikitext", config_name="wikitext-103-raw-v1", split="train", tokenizer=tokenizer)

    # 使用 DataLoader 进行数据加载
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 打印一些示例数据
    for batch in dataloader:
        print(batch)
        break
