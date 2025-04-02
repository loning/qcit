import torch
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
from mpl_toolkits.mplot3d import Axes3D
import random

def generate_control_tensor(style="normal", task="classification", intensity=0.5, dim=64):
    """生成控制张量
    
    Args:
        style: 控制风格，可选 "normal", "creative", "focused"
        task: 任务类型，可选 "classification", "generation", "translation"
        intensity: 控制强度，范围 [0, 1]
        dim: 控制向量维度
        
    Returns:
        控制向量张量
    """
    # 初始化控制向量
    control = torch.zeros(dim)
    
    # 风格编码 (前1/3维度)
    style_idx = dim // 3
    if style == "normal":
        control[:style_idx] = torch.randn(style_idx) * 0.1
    elif style == "creative":
        control[:style_idx] = torch.randn(style_idx) * 0.5 + 0.5
    elif style == "focused":
        control[:style_idx] = -torch.abs(torch.randn(style_idx) * 0.5) - 0.2
    
    # 任务编码 (中间1/3维度)
    task_idx = 2 * (dim // 3)
    if task == "classification":
        control[style_idx:task_idx] = torch.ones(task_idx-style_idx) * 0.8
    elif task == "generation":
        control[style_idx:task_idx] = torch.sin(torch.linspace(0, 3*np.pi, task_idx-style_idx)) * 0.5 + 0.5
    elif task == "translation":
        control[style_idx:task_idx] = torch.cos(torch.linspace(0, 2*np.pi, task_idx-style_idx)) * 0.5 + 0.5
    
    # 强度编码 (最后1/3维度)
    control[task_idx:] = torch.ones(dim-task_idx) * intensity
    
    return control

def visualize_trajectory(trajectory, title="QCIT Trajectory Evolution", filename="qcit_trajectory.png"):
    """可视化轨迹演化
    
    Args:
        trajectory: 状态轨迹列表 [tensor1, tensor2, ...]
        title: 图表标题
        filename: 保存的图像文件名
    """
    if not trajectory:
        print("Trajectory is empty, cannot visualize")
        return
    
    # 计算主成分分析维度降维（简化版PCA）
    states = torch.stack([state.mean(0) for state in trajectory])
    states_np = states.detach().cpu().numpy()
    
    # 轨迹长度
    T = states_np.shape[0]
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 绘制状态演化热图
    plt.subplot(2, 2, 1)
    plt.imshow(states_np, aspect='auto', cmap='viridis')
    plt.colorbar(label='State Activation')
    plt.title('State Evolution Heatmap')
    plt.xlabel('Dimension')
    plt.ylabel('Time Step')
    
    # 绘制状态方差曲线
    plt.subplot(2, 2, 2)
    variances = [torch.var(state).item() for state in trajectory]
    plt.plot(variances, 'r-', linewidth=2)
    plt.title('State Variance Curve')
    plt.xlabel('Time Step')
    plt.ylabel('Variance')
    
    # 绘制轨迹前三维演化
    if states_np.shape[1] >= 3:
        plt.subplot(2, 2, 3)
        for i in range(T-1):
            plt.plot(states_np[i:i+2, 0], states_np[i:i+2, 1], 'b-', alpha=0.5*(i+1)/T, linewidth=2)
        plt.scatter(states_np[:, 0], states_np[:, 1], c=range(T), cmap='viridis')
        plt.title('First Two Dimensions Trajectory')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        
        if T >= 3:
            plt.subplot(2, 2, 4)
            fig = plt.gcf()
            ax = fig.add_subplot(2, 2, 4, projection='3d')
            for i in range(T-1):
                ax.plot(states_np[i:i+2, 0], states_np[i:i+2, 1], states_np[i:i+2, 2], 
                        'b-', alpha=0.5*(i+1)/T, linewidth=2)
            scatter = ax.scatter(states_np[:, 0], states_np[:, 1], states_np[:, 2], 
                       c=range(T), cmap='viridis')
            plt.colorbar(scatter, label='Time Step')
            ax.set_title('First Three Dimensions Trajectory')
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.set_zlabel('Dimension 3')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    print(f"Trajectory visualization saved as '{filename}'")

def get_complexity_analysis():
    """返回复杂度分析"""
    analysis = {
        "SRCO": {"时间复杂度": "O(n·d)", "空间复杂度": "O(n·d)"},
        "MRAO": {"时间复杂度": "O(W·d)", "空间复杂度": "O(W·d)"},
        "控制生成": {"时间复杂度": "O(d²)", "空间复杂度": "O(d²)"},
        "偏好打分": {"时间复杂度": "O(T·d)", "空间复杂度": "O(T·d)"}
    }
    return analysis 