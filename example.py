import torch
from transformers import BertTokenizer
from qcit.models import QCITModel
from qcit.utils import generate_control_tensor, visualize_trajectory, get_complexity_analysis

def main():
    # 初始化模型
    print("初始化QCIT+模型...")
    model = QCITModel(dim=768, control_dim=64, window_size=5)
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    
    # 准备输入文本
    text = "这是一个测试"
    print(f"输入文本: {text}")
    
    # 编码文本
    inputs = tokenizer(text, return_tensors="pt")
    
    # 初始化轨迹
    print("初始化状态轨迹...")
    output, score = model(inputs, modality='text')
    print(f"输出维度: {output.shape}, 方差分数: {score}")
    
    # 生成不同风格的控制张量
    control_signals = [
        generate_control_tensor(style="normal", task="classification", intensity=0.3),
        generate_control_tensor(style="creative", task="generation", intensity=0.7),
        generate_control_tensor(style="focused", task="translation", intensity=0.5)
    ]
    
    print("\n生成不同控制信号下的演化轨迹...")
    trajectories = []
    
    # 正常风格演化
    print("\n1. 使用'normal'风格控制信号演化...")
    outputs, scores = model.evolve_trajectory(steps=5, control_signals=[control_signals[0]]*5)
    print(f"  最终状态方差: {scores[-1].item():.4f}")
    trajectories.append(model.trajectory_history.copy())
    
    # 重置模型并用创造性风格演化
    model = QCITModel(dim=768, control_dim=64, window_size=5)
    output, score = model(inputs, modality='text')
    print("\n2. 使用'creative'风格控制信号演化...")
    outputs, scores = model.evolve_trajectory(steps=5, control_signals=[control_signals[1]]*5)
    print(f"  最终状态方差: {scores[-1].item():.4f}")
    trajectories.append(model.trajectory_history.copy())
    
    # 重置模型并用专注风格演化
    model = QCITModel(dim=768, control_dim=64, window_size=5)
    output, score = model(inputs, modality='text')
    print("\n3. 使用'focused'风格控制信号演化...")
    outputs, scores = model.evolve_trajectory(steps=5, control_signals=[control_signals[2]]*5)
    print(f"  最终状态方差: {scores[-1].item():.4f}")
    trajectories.append(model.trajectory_history.copy())
    
    # 可视化三种轨迹
    print("\n可视化轨迹演化...")
    for i, trajectory in enumerate(trajectories):
        style_names = ["Normal", "Creative", "Focused"]
        visualize_trajectory(
            trajectory, 
            f"QCIT+ {style_names[i]} Style Trajectory Evolution", 
            f"qcit_{style_names[i].lower()}_trajectory.png"
        )
    
    # 输出复杂度分析
    complexity = get_complexity_analysis()
    print("\n模型复杂度分析:")
    for component, analysis in complexity.items():
        print(f"  {component}: 时间复杂度={analysis['时间复杂度']}, 空间复杂度={analysis['空间复杂度']}")
    
    print("\nQCIT+模型优化模块演示完成!")

if __name__ == "__main__":
    main() 