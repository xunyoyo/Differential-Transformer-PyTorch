# data/tokenizer.py

import tiktoken

class Tokenizer:
    """
    使用 tiktoken 的 cl100k_base 分词器的封装类。
    负责将文本转换为 token IDs，以及将 token IDs 转换回文本。
    """
    def __init__(self):
        # 初始化分词器，使用 "cl100k_base" 编码
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def encode(self, text):
        """
        将文本编码为 token ID 列表。

        Args:
            text (str): 要分词的输入文本。

        Returns:
            list[int]: 代表 token ID 的整数列表。
        """
        return self.tokenizer.encode(text)

    def decode(self, token_ids):
        """
        将 token ID 列表解码为文本。

        Args:
            token_ids (list[int]): token ID 的整数列表。

        Returns:
            str: 解码后的文本字符串。
        """
        return self.tokenizer.decode(token_ids)

    def batch_encode(self, texts):
        """
        批量编码多个文本。

        Args:
            texts (list[str]): 要编码的文本列表。

        Returns:
            list[list[int]]: 每个文本被编码为 token ID 列表的列表。
        """
        return [self.encode(text) for text in texts]

# 测试部分（如果需要，直接运行该脚本进行测试）
if __name__ == "__main__":
    tokenizer = Tokenizer()
    
    # 示例文本
    text = "This is a test sentence for tokenization."

    # 编码文本
    token_ids = tokenizer.encode(text)
    print(f"Encoded: {token_ids}")

    # 解码 token ID
    decoded_text = tokenizer.decode(token_ids)
    print(f"Decoded: {decoded_text}")

    # 批量编码
    texts = ["This is the first sentence.", "This is the second sentence."]
    batch_token_ids = tokenizer.batch_encode(texts)
    print(f"Batch Encoded: {batch_token_ids}")
