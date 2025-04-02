#!/usr/bin/env python
"""
QCIT+基准测试运行脚本
"""

import argparse
import torch
from benchmarks.run_benchmarks import run_sst2_benchmarks, run_wikitext_benchmarks

def main():
    parser = argparse.ArgumentParser(description="运行QCIT+模型基准测试")
    parser.add_argument("--output_dir", type=str, default="benchmark_results", help="结果输出目录")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--tasks", type=str, default="all", help="要运行的任务: 'sst2', 'wikitext', 或 'all'")
    parser.add_argument("--gpu", action="store_true", help="是否使用GPU（如可用）")
    
    args = parser.parse_args()
    
    # 检测可用的GPU设备
    gpu_info = "不使用"
    if args.gpu:
        if torch.cuda.is_available():
            gpu_info = f"NVIDIA GPU ({torch.cuda.get_device_name(0)})"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu_info = "Apple Silicon GPU (MPS)"
        else:
            gpu_info = "无可用GPU，将使用CPU"
    
    print("="*70)
    print("  QCIT+模型基准测试 - 与BERT、Standard和GPT-2的性能比较  ")
    print("="*70)
    print(f"• 输出目录: {args.output_dir}")
    print(f"• 批次大小: {args.batch_size}")
    print(f"• 训练轮数: {args.epochs}")
    print(f"• 测试任务: {args.tasks}")
    print(f"• GPU加速: {gpu_info}")
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