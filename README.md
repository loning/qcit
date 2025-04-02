# 量子经典同构变换器 (QCIT+)

量子经典同构变换器（Quantum Classical Isomorphic Transformer, QCIT+）是一种基于量子经典自参照宇宙（QCSU）理论的先进深度学习架构。本项目实现了QCIT+的优化模块，将量子状态处理与经典深度学习相结合。

## 🌟 特性

- **多模态融合**：通过模态桥接编码器统一处理文本、图像和数值数据
- **轨迹演化**：使用扩展MRAO轨迹算子实现有窗口的无限维度递归轨迹调控
- **控制张量**：支持目标/风格注入，实现对生成过程的精确控制
- **结构偏好**：通过结构轨迹偏好打分函数评估演化过程的熵激活密度与稳定性

## 📋 项目结构

```
QCIT/
├── qcit/
│   ├── __init__.py           # 包初始化文件
│   ├── models.py             # 核心模型实现
│   └── utils.py              # 工具函数（控制张量生成、轨迹可视化等）
├── example.py                # 使用示例
└── requirements.txt          # 项目依赖
```

## 🔧 安装

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/QCIT.git
cd QCIT
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 🚀 快速开始

运行示例脚本：

```bash
python example.py
```

示例脚本将：
1. 初始化QCIT+模型
2. 使用示例文本初始化状态轨迹
3. 应用不同风格的控制张量生成演化轨迹
4. 可视化不同轨迹
5. 输出模型复杂度分析

## 📐 核心模块

### 模型扩展元组

原始模型定义为：
```
QCIT = ⟨ Q, C, SRCO, MRAO, QCDAO ⟩
```

扩展模型定义为：
```
QCIT+ = ⟨ Q, C, SRCO, MRAO*, QCDAO, B_init, H, G, L_pref, T_ctrl ⟩
```

### 模块详解

1. **经典态初始化（BERT）**：将文本映射到初始经典表示
2. **模态桥接编码器**：统一处理各种模态的输入
3. **扩展MRAO轨迹算子**：通过历史窗口实现轨迹演化
4. **结构偏好打分函数**：评估轨迹演化质量
5. **控制张量生成器**：注入控制信号调整演化方向

## 📊 复杂度分析

| 组件 | 时间复杂度 | 空间复杂度 |
|------|------------|------------|
| SRCO | O(n·d) | O(n·d) |
| MRAO | O(W·d) | O(W·d) |
| 控制生成 | O(d²) | O(d²) |
| 偏好打分 | O(T·d) | O(T·d) |

系统保持线性可控，支持多模态与控制指令动态调节。

## 📝 引用

如果您在研究中使用QCIT+，请引用：

```
@article{qcit2023,
  title={Quantum Classical Isomorphic Transformer: A Novel Approach to Multi-Modal Learning},
  author={Your Name},
  journal={Arxiv},
  year={2023}
}
```

## �� 许可证

MIT License 