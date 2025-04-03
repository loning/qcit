#!/usr/bin/env python
"""
QCIT+基准测试运行脚本
"""

import argparse
import torch
import sys
from benchmarks.run_benchmarks import run_sst2_benchmarks, run_wikitext_benchmarks

def main():
    parser = argparse.ArgumentParser(description="运行QCIT+模型基准测试")
    parser.add_argument("--output_dir", type=str, default="benchmark_results", help="结果输出目录")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--tasks", type=str, default="all", help="要运行的任务: 'sst2', 'wikitext', 或 'all'")
    parser.add_argument("--gpu", action="store_true", help="是否使用GPU（如可用）")
    parser.add_argument("--skip_training", action="store_true", help="跳过训练过程，只运行评估")
    
    args = parser.parse_args()
    
    # 检测可用的GPU设备
    gpu_info = "不使用"
    device = "cpu"  # 默认使用CPU
    
    if args.gpu:
        print("正在检测GPU设备...")
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            gpu_info = f"NVIDIA GPU ({device_name}, 共{device_count}个设备)"
            print(f"✓ 检测到NVIDIA GPU: {device_name}")
            print(f"✓ CUDA版本: {torch.version.cuda}")
            print(f"✓ PyTorch版本: {torch.__version__}")
            device = "cuda"  # 使用CUDA设备
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu_info = "Apple Silicon GPU (MPS)"
            print("✓ 检测到Apple Silicon GPU (MPS)")
            device = "mps"  # 使用Apple Silicon GPU
        else:
            gpu_info = "无可用GPU，将使用CPU"
            print("✗ 未检测到可用GPU，将使用CPU")
            print("  可能的原因:")
            print("  - 未安装CUDA或CUDA版本不兼容")
            print("  - 未安装NVIDIA驱动程序")
            print("  - 系统没有NVIDIA GPU或Apple Silicon芯片")
            print("  - PyTorch未编译GPU支持")
    
    # 添加device属性到args对象
    args.device = device
    
    print("="*70)
    print("  QCIT+模型基准测试 - 与BERT、Standard和GPT-2的性能比较  ")
    print("="*70)
    print(f"• 输出目录: {args.output_dir}")
    print(f"• 批次大小: {args.batch_size}")
    print(f"• 训练轮数: {args.epochs}")
    print(f"• 测试任务: {args.tasks}")
    print(f"• GPU加速: {gpu_info}")
    print(f"• 使用设备: {device}")
    if args.skip_training:
        print(f"• 跳过训练: 是")
    print("-"*70)
    
    # 运行指定任务的基准测试
    if args.tasks.lower() in ["all", "sst2"]:
        sst2_metrics = run_sst2_benchmarks(args)
    
    if args.tasks.lower() in ["all", "wikitext"]:
        wikitext_metrics = run_wikitext_benchmarks(args)
    
    print("\n" + "="*70)
    print("所有基准测试完成！输出文件已保存到 {}".format(args.output_dir))
    print("="*70)

if __name__ == "__main__":
    main() 