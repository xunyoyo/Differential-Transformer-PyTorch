# scripts/train.py

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup  # 用于实现 warmup 策略

from data.dataset import HuggingFaceDataset
from data.tokenizer import Tokenizer
from DiffTransformer import DiffTransformer

# 超参数设置
vocab_size = 100288
d_model = 3072
num_heads = 24
num_layers = 28
max_seq_length = 512
batch_size = 8
learning_rate = 3.2e-4
weight_decay = 0.1
warmup_steps = 1000
epochs = 10

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化分词器和数据集
tokenizer = Tokenizer()
dataset = HuggingFaceDataset(dataset_name="wikitext", config_name="wikitext-103-raw-v1", split="train", tokenizer=tokenizer, max_length=max_seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型
model = DiffTransformer(vocab_size, d_model, num_heads, num_layers, max_seq_length)
model.to(device)

# 初始化优化器
optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=weight_decay)

# 设置学习率调度器 (带有 warmup)
total_steps = len(dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 训练循环
model.train()
for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        # 将 batch 数据转移到 GPU 上
        input_ids = batch['input_ids'].to(device)

        # 输入和标签处理
        labels = input_ids[:, 1:].contiguous()  # 标签为输入的右移版本
        input_ids = input_ids[:, :-1].contiguous()  # 输入为去掉最后一个 token 的部分

        # 前向传播
        outputs = model(input_ids)

        # 计算损失
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()  # 更新学习率

        # 打印损失
        if step % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Step [{step}/{len(dataloader)}], Loss: {loss.item()}")

    # 保存模型检查点
    checkpoint_path = f"../checkpoints/diff_transformer_epoch_{epoch + 1}.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved at {checkpoint_path}")

print("Training complete.")
