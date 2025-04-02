"""
主要基准测试运行脚本：用于执行所有基准测试
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd

# 导入自定义模块
from benchmarks.data import get_dataloader
from benchmarks.models import get_model, count_parameters
from benchmarks.training import Trainer, LanguageModelTrainer, evaluate_classifier, evaluate_language_model
from benchmarks.report import save_metrics_to_json, generate_comparison_table, generate_pdf_report

def run_sst2_benchmarks(args):
    """
    在SST-2数据集上运行分类任务基准测试
    
    参数:
        args: 命令行参数
    """
    print("\n" + "="*50)
    print("开始在SST-2数据集上运行分类任务基准测试")
    print("="*50 + "\n")
    
    # 设置输出目录
    output_dir = os.path.join(args.output_dir, "sst2")
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print("加载SST-2数据...")
    train_dataloader = get_dataloader("sst2", "train", batch_size=args.batch_size)
    val_dataloader = get_dataloader("sst2", "validation", batch_size=args.batch_size)
    test_dataloader = get_dataloader("sst2", "validation", batch_size=args.batch_size)  # 使用验证集作为测试集
    
    # 设置要测试的模型
    model_configs = [
        ("bert", {}),
        ("standard", {}),
        ("gpt2", {}),
        ("qcit", {})
    ]
    
    # 存储所有模型的指标
    all_metrics = {}
    
    # 循环测试每个模型
    for model_name, model_kwargs in model_configs:
        print(f"\n测试模型: {model_name}")
        
        # 初始化模型
        model = get_model(model_name, task="classification", **model_kwargs)
        
        # 计算参数量
        num_params = count_parameters(model)
        print(f"模型参数量: {num_params:,}")
        
        # 设置训练器
        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=args.epochs,
            log_dir=os.path.join(output_dir, "logs"),
            model_dir=os.path.join(output_dir, "models"),
            model_name=model_name,
            device=args.device
        )
        
        # 训练模型
        train_metrics = trainer.train()
        trainer.close()
        
        # 在测试集上评估
        test_metrics = evaluate_classifier(
            model=model,
            test_dataloader=test_dataloader,
            device=args.device
        )
        
        # 合并指标
        metrics = {**train_metrics, **test_metrics, "parameters": num_params}
        all_metrics[model_name] = metrics
        
        # 保存单个模型的指标
        model_output_path = os.path.join(output_dir, f"{model_name}_metrics.json")
        save_metrics_to_json(metrics, model_output_path)
    
    # 保存所有模型的综合指标
    combined_output_path = os.path.join(output_dir, "all_metrics.json")
    save_metrics_to_json(all_metrics, combined_output_path)
    
    # 生成比较表格
    comparison_table_path = os.path.join(output_dir, "comparison_table.csv")
    generate_comparison_table(all_metrics, task_type="classification", output_path=comparison_table_path)
    
    # 生成PDF报告
    pdf_path = os.path.join(output_dir, "sst2_benchmark_report.pdf")
    generate_pdf_report(
        metrics=all_metrics,
        task_type="classification",
        output_path=pdf_path,
        dataset_name="SST-2"
    )
    
    print(f"\nSST-2基准测试完成。报告已保存到: {pdf_path}")
    return all_metrics

def run_wikitext_benchmarks(args):
    """
    在WikiText数据集上运行语言建模任务基准测试
    
    参数:
        args: 命令行参数
    """
    print("\n" + "="*50)
    print("开始在WikiText数据集上运行语言建模任务基准测试")
    print("="*50 + "\n")
    
    # 设置输出目录
    output_dir = os.path.join(args.output_dir, "wikitext")
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print("加载WikiText数据...")
    train_dataloader = get_dataloader("wikitext", "train", batch_size=args.batch_size, tokenizer_type="gpt2")
    val_dataloader = get_dataloader("wikitext", "validation", batch_size=args.batch_size, tokenizer_type="gpt2")
    test_dataloader = get_dataloader("wikitext", "test", batch_size=args.batch_size, tokenizer_type="gpt2")
    
    # 设置要测试的模型
    model_configs = [
        ("gpt2", {}),  # GPT-2是主要的语言模型
        ("bert", {"is_decoder": True}),  # 将BERT作为解码器使用
        ("qcit", {})  # QCIT模型
    ]
    
    # 存储所有模型的指标
    all_metrics = {}
    
    # 循环测试每个模型
    for model_name, model_kwargs in model_configs:
        if model_name == "standard":
            print(f"\n跳过模型: {model_name} - 不支持语言建模任务")
            continue
            
        print(f"\n测试模型: {model_name}")
        
        try:
            # 初始化模型
            model = get_model(model_name, task="language_modeling", **model_kwargs)
            
            # 计算参数量
            num_params = count_parameters(model)
            print(f"模型参数量: {num_params:,}")
            
            # 设置训练器
            trainer = LanguageModelTrainer(
                model=model,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                num_epochs=args.epochs,
                log_dir=os.path.join(output_dir, "logs"),
                model_dir=os.path.join(output_dir, "models"),
                model_name=model_name,
                device=args.device
            )
            
            # 训练模型
            train_metrics = trainer.train()
            trainer.close()
            
            # 在测试集上评估
            test_metrics = evaluate_language_model(
                model=model,
                test_dataloader=test_dataloader,
                device=args.device
            )
            
            # 合并指标
            metrics = {**train_metrics, **test_metrics, "parameters": num_params}
            all_metrics[model_name] = metrics
            
            # 保存单个模型的指标
            model_output_path = os.path.join(output_dir, f"{model_name}_metrics.json")
            save_metrics_to_json(metrics, model_output_path)
        
        except NotImplementedError as e:
            print(f"跳过模型: {model_name} - {str(e)}")
    
    # 保存所有模型的综合指标
    combined_output_path = os.path.join(output_dir, "all_metrics.json")
    save_metrics_to_json(all_metrics, combined_output_path)
    
    # 生成比较表格
    comparison_table_path = os.path.join(output_dir, "comparison_table.csv")
    generate_comparison_table(all_metrics, task_type="language_modeling", output_path=comparison_table_path)
    
    # 生成PDF报告
    pdf_path = os.path.join(output_dir, "wikitext_benchmark_report.pdf")
    generate_pdf_report(
        metrics=all_metrics,
        task_type="language_modeling",
        output_path=pdf_path,
        dataset_name="WikiText"
    )
    
    print(f"\nWikiText基准测试完成。报告已保存到: {pdf_path}")
    return all_metrics

def main():
    parser = argparse.ArgumentParser(description="运行QCIT+模型基准测试")
    parser.add_argument("--output_dir", type=str, default="benchmark_results", help="结果输出目录")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--tasks", type=str, default="all", help="要运行的任务: 'sst2', 'wikitext', 或 'all'")
    parser.add_argument("--gpu", action="store_true", help="是否使用GPU（如可用）")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备 - 增加对MPS (Apple Silicon GPU)的支持
    if args.gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"使用NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("使用Apple Silicon GPU (MPS)")
        else:
            device = torch.device("cpu")
            print("没有可用的GPU，使用CPU")
    else:
        device = torch.device("cpu")
        print("使用CPU")
    
    args.device = device
    
    # 运行指定任务的基准测试
    if args.tasks.lower() in ["all", "sst2"]:
        sst2_metrics = run_sst2_benchmarks(args)
    
    if args.tasks.lower() in ["all", "wikitext"]:
        wikitext_metrics = run_wikitext_benchmarks(args)
    
    print("\n所有基准测试完成！")

if __name__ == "__main__":
    main() 