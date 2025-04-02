"""
模型模块：用于加载和封装用于基准测试的各种模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertForSequenceClassification
from transformers import GPT2Model, GPT2LMHeadModel
from transformers import BertTokenizer, GPT2Tokenizer
from qcit.models import QCITModel

class BERTClassifier(nn.Module):
    """基于BERT的分类器"""
    
    def __init__(self, num_classes=2, pretrained_model="bert-base-chinese"):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_token)
        return logits

class StandardClassifier(nn.Module):
    """标准Transformer编码器分类器"""
    
    def __init__(self, vocab_size=21128, hidden_size=768, num_layers=6, num_heads=8, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size*4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        
        if attention_mask is not None:
            # 创建一个掩码，以便填充词元被屏蔽
            # 为TransformerEncoder转换掩码，True表示需要掩码的位置
            src_key_padding_mask = (attention_mask == 0)
            x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        else:
            x = self.encoder(x)
        
        # 使用[CLS]标记的输出进行分类
        logits = self.classifier(x[:, 0])
        return logits

class GPT2Classifier(nn.Module):
    """基于GPT-2的分类器"""
    
    def __init__(self, num_classes=2, pretrained_model="gpt2"):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained(pretrained_model)
        self.classifier = nn.Linear(self.gpt2.config.n_embd, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        # 使用最后一个非填充位置的隐藏状态
        last_hidden = outputs.last_hidden_state
        
        # 使用序列的最后一个token
        if attention_mask is not None:
            # 获取每个序列中最后一个非填充token的位置
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = input_ids.size(0)
            last_token_hidden = torch.stack([
                last_hidden[i, seq_lengths[i]] for i in range(batch_size)
            ])
        else:
            # 如果没有掩码，就使用最后一个token
            last_token_hidden = last_hidden[:, -1]
        
        logits = self.classifier(last_token_hidden)
        return logits

class QCITClassifier(nn.Module):
    """基于QCIT的分类器"""
    
    def __init__(self, num_classes=2, dim=768, control_dim=64, window_size=5):
        super().__init__()
        self.qcit = QCITModel(dim=dim, control_dim=control_dim, window_size=window_size)
        self.classifier = nn.Linear(dim, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        # 准备输入
        inputs = {"input_ids": input_ids}
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask
        
        # 获取QCIT输出和分数
        output, _ = self.qcit(inputs, modality='text', initialize=True)
        
        # 分类
        logits = self.classifier(output)
        return logits

def get_model(model_name, task="classification", **kwargs):
    """
    根据模型名称和任务类型返回相应的模型实例
    
    参数:
        model_name: 模型名称，'bert', 'standard', 'gpt2', 'qcit'
        task: 任务类型，'classification' 或 'language_modeling'
        **kwargs: 其他传递给模型构造函数的参数
    """
    if task == "classification":
        if model_name.lower() == "bert":
            return BERTClassifier(**kwargs)
        elif model_name.lower() == "standard":
            return StandardClassifier(**kwargs)
        elif model_name.lower() == "gpt2":
            return GPT2Classifier(**kwargs)
        elif model_name.lower() == "qcit":
            return QCITClassifier(**kwargs)
        else:
            raise ValueError(f"不支持的模型: {model_name}")
    
    elif task == "language_modeling":
        if model_name.lower() == "bert":
            # 使用BERT作为掩码语言模型
            return BertForSequenceClassification.from_pretrained("bert-base-chinese", **kwargs)
        elif model_name.lower() == "standard":
            # 实现自定义标准语言模型...
            raise NotImplementedError("Standard语言模型尚未实现")
        elif model_name.lower() == "gpt2":
            return GPT2LMHeadModel.from_pretrained("gpt2", **kwargs)
        elif model_name.lower() == "qcit":
            # 将QCIT适配为语言模型...
            # 这里需要扩展QCITModel以支持语言建模任务
            qcit = QCITModel(**kwargs)
            # 添加语言模型头
            lm_head = nn.Linear(kwargs.get("dim", 768), 21128)  # 使用与BERT相同的词汇表大小
            model = nn.Sequential(qcit, lm_head)
            return model
        else:
            raise ValueError(f"不支持的模型: {model_name}")
    
    else:
        raise ValueError(f"不支持的任务类型: {task}")

def count_parameters(model):
    """
    计算模型的可训练参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 