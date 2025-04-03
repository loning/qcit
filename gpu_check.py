import torch
import sys

def check_gpu():
    print("Python版本:", sys.version)
    print("PyTorch版本:", torch.__version__)
    print("\nGPU信息:")
    
    if torch.cuda.is_available():
        print(f"CUDA是否可用: 是")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"GPU {i} 显存总量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"GPU {i} 当前显存使用: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"GPU {i} 当前显存缓存: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
    else:
        print("CUDA是否可用: 否")
        print("未检测到可用的GPU")

if __name__ == "__main__":
    check_gpu() 