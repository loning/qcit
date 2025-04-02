"""
训练和评估模块：用于模型的训练和性能评估
"""

import os
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    """模型训练器类"""
    
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        criterion=None,
        optimizer=None,
        scheduler=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_epochs=3,
        log_dir="logs",
        model_dir="models",
        model_name="model"
    ):
        """
        初始化训练器
        
        参数:
            model: 要训练的模型
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            criterion: 损失函数，如果为None则使用交叉熵损失
            optimizer: 优化器，如果为None则使用AdamW
            scheduler: 学习率调度器，可选
            device: 训练设备
            num_epochs: 训练轮数
            log_dir: 日志目录
            model_dir: 模型保存目录
            model_name: 模型名称
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.num_epochs = num_epochs
        self.model_dir = model_dir
        self.model_name = model_name
        
        # 确保目录存在
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # 设置tensorboard
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, model_name))
        
        # 如果未提供，则设置默认损失函数和优化器
        self.criterion = criterion if criterion else nn.CrossEntropyLoss()
        self.optimizer = optimizer if optimizer else optim.AdamW(model.parameters(), lr=2e-5)
        self.scheduler = scheduler
        
        # 将模型移至指定设备
        self.model.to(self.device)
        
        # 存储训练过程中的指标
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.training_time = 0
        self.inference_time = 0
    
    def train_epoch(self, epoch):
        """
        训练一个完整的轮次
        """
        self.model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        num_batches = len(self.train_dataloader)
        
        for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            start_time = time.time()
            logits = self.model(input_ids, attention_mask)
            
            # 计算损失
            loss = self.criterion(logits, labels)
            
            # 反向传播和优化
            loss.backward()
            self.optimizer.step()
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
            
            # 计算准确率
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == labels).float().mean()
            
            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()
        
        # 计算平均损失和准确率
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        
        # 记录到tensorboard
        self.writer.add_scalar("Loss/train", avg_loss, epoch)
        self.writer.add_scalar("Accuracy/train", avg_accuracy, epoch)
        
        # 存储指标
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(avg_accuracy)
        
        return avg_loss, avg_accuracy
    
    def validate(self, epoch):
        """
        在验证集上评估模型
        """
        self.model.eval()
        epoch_loss = 0
        epoch_accuracy = 0
        num_batches = len(self.val_dataloader)
        
        inference_start = time.time()
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                
                # 前向传播
                logits = self.model(input_ids, attention_mask)
                
                # 计算损失
                loss = self.criterion(logits, labels)
                
                # 计算准确率
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == labels).float().mean()
                
                epoch_loss += loss.item()
                epoch_accuracy += accuracy.item()
        
        inference_end = time.time()
        self.inference_time = inference_end - inference_start
        
        # 计算平均损失和准确率
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        
        # 记录到tensorboard
        self.writer.add_scalar("Loss/val", avg_loss, epoch)
        self.writer.add_scalar("Accuracy/val", avg_accuracy, epoch)
        
        # 存储指标
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(avg_accuracy)
        
        return avg_loss, avg_accuracy
    
    def train(self):
        """
        训练模型指定的轮次
        """
        print(f"开始在{self.device}上训练{self.model_name}...")
        
        best_val_accuracy = 0
        total_start_time = time.time()
        
        for epoch in range(self.num_epochs):
            # 训练一个轮次
            train_loss, train_accuracy = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_accuracy = self.validate(epoch)
            
            # 打印进度
            print(f"Epoch {epoch+1}/{self.num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            
            # 保存最佳模型
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                model_path = os.path.join(self.model_dir, f"{self.model_name}_best.pt")
                torch.save(self.model.state_dict(), model_path)
                print(f"  保存最佳模型到 {model_path}")
        
        total_end_time = time.time()
        self.training_time = total_end_time - total_start_time
        
        print(f"训练完成！总时间: {self.training_time:.2f}秒")
        
        # 保存最终模型
        model_path = os.path.join(self.model_dir, f"{self.model_name}_final.pt")
        torch.save(self.model.state_dict(), model_path)
        print(f"最终模型已保存到 {model_path}")
        
        return {
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
            "train_accuracy": self.train_accuracies,
            "val_accuracy": self.val_accuracies,
            "training_time": self.training_time,
            "inference_time": self.inference_time,
            "best_val_accuracy": best_val_accuracy
        }
    
    def close(self):
        """
        关闭tensorboard writer
        """
        self.writer.close()

class LanguageModelTrainer(Trainer):
    """专用于语言模型训练的训练器"""
    
    def train_epoch(self, epoch):
        """
        训练语言模型一个完整的轮次
        """
        self.model.train()
        epoch_loss = 0
        epoch_perplexity = 0
        num_batches = len(self.train_dataloader)
        
        for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # 反向传播和优化
            loss.backward()
            self.optimizer.step()
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
            
            # 计算困惑度
            perplexity = torch.exp(loss)
            
            epoch_loss += loss.item()
            epoch_perplexity += perplexity.item()
        
        # 计算平均损失和困惑度
        avg_loss = epoch_loss / num_batches
        avg_perplexity = epoch_perplexity / num_batches
        
        # 记录到tensorboard
        self.writer.add_scalar("Loss/train", avg_loss, epoch)
        self.writer.add_scalar("Perplexity/train", avg_perplexity, epoch)
        
        # 存储指标
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(avg_perplexity)  # 重用此字段存储困惑度
        
        return avg_loss, avg_perplexity
    
    def validate(self, epoch):
        """
        在验证集上评估语言模型
        """
        self.model.eval()
        epoch_loss = 0
        epoch_perplexity = 0
        num_batches = len(self.val_dataloader)
        
        inference_start = time.time()
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # 前向传播
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                # 计算困惑度
                perplexity = torch.exp(loss)
                
                epoch_loss += loss.item()
                epoch_perplexity += perplexity.item()
        
        inference_end = time.time()
        self.inference_time = inference_end - inference_start
        
        # 计算平均损失和困惑度
        avg_loss = epoch_loss / num_batches
        avg_perplexity = epoch_perplexity / num_batches
        
        # 记录到tensorboard
        self.writer.add_scalar("Loss/val", avg_loss, epoch)
        self.writer.add_scalar("Perplexity/val", avg_perplexity, epoch)
        
        # 存储指标
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(avg_perplexity)  # 重用此字段存储困惑度
        
        return avg_loss, avg_perplexity

def evaluate_classifier(model, test_dataloader, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    评估分类器在测试集上的性能
    
    返回：
        准确率，精确率，召回率，F1分数，推理时间
    """
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    inference_start = time.time()
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    inference_end = time.time()
    inference_time = inference_end - inference_start
    
    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average="binary"
    )
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "inference_time": inference_time
    }

def evaluate_language_model(model, test_dataloader, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    评估语言模型在测试集上的性能
    
    返回：
        困惑度，损失，推理时间
    """
    model.to(device)
    model.eval()
    
    total_loss = 0
    total_tokens = 0
    
    inference_start = time.time()
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # 计算当前批次的tokens数量
            batch_tokens = attention_mask.sum().item()
            
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
    
    inference_end = time.time()
    inference_time = inference_end - inference_start
    
    # 计算平均损失和困惑度
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return {
        "perplexity": perplexity,
        "loss": avg_loss,
        "inference_time": inference_time
    } 