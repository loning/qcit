"""
数据处理模块：用于加载并处理SST-2和WikiText数据集
"""

import os
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, GPT2Tokenizer

class SST2Dataset(Dataset):
    """情感分析数据集包装器"""
    
    def __init__(self, split="train", max_length=128):
        """
        初始化SST-2数据集
        
        参数:
            split: 数据分割（'train', 'validation', 'test'）
            max_length: 序列的最大长度
        """
        self.dataset = load_dataset("glue", "sst2", split=split)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item["sentence"]
        label = item["label"]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 移除批次维度
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "label": torch.tensor(label, dtype=torch.long)
        }

class WikiTextDataset(Dataset):
    """WikiText语言建模数据集包装器"""
    
    def __init__(self, split="train", max_length=256, tokenizer_type="bert"):
        """
        初始化WikiText数据集
        
        参数:
            split: 数据分割（'train', 'validation', 'test'）
            max_length: 序列的最大长度
            tokenizer_type: 使用的分词器类型 ('bert' 或 'gpt2')
        """
        self.dataset = load_dataset("wikitext", "wikitext-103-v1", split=split)
        
        if tokenizer_type == "bert":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        elif tokenizer_type == "gpt2":
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            raise ValueError(f"不支持的分词器类型: {tokenizer_type}")
            
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item["text"]
        
        # 跳过空文本
        if not text.strip():
            # 使用下一个非空文本，如果已经是最后一个，则回到开始
            next_idx = (idx + 1) % len(self.dataset)
            return self.__getitem__(next_idx)
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 移除批次维度
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # 对于语言建模任务，输入序列为目标序列
        labels = encoding["input_ids"].clone()
        
        # 为BERT掩码语言建模设置15%的标记为掩码（如果需要）
        
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": labels
        }

def get_dataloader(dataset_name, split, batch_size=32, tokenizer_type="bert"):
    """
    根据数据集名称返回相应的数据加载器
    
    参数:
        dataset_name: 数据集名称，'sst2' 或 'wikitext'
        split: 数据分割，'train'、'validation' 或 'test'
        batch_size: 批次大小
        tokenizer_type: 分词器类型（对于WikiText数据集）
    """
    if dataset_name.lower() == "sst2":
        dataset = SST2Dataset(split=split)
    elif dataset_name.lower() == "wikitext":
        dataset = WikiTextDataset(split=split, tokenizer_type=tokenizer_type)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=2,
        pin_memory=True
    ) 