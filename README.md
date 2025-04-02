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
├── benchmarks/               # 基准测试框架
│   ├── __init__.py           # 测试包初始化文件
│   ├── data.py               # 数据加载和处理
│   ├── models.py             # 模型实现和管理
│   ├── training.py           # 训练和评估
│   ├── report.py             # 报告生成
│   └── run_benchmarks.py     # 基准测试运行
├── run_benchmarks.py         # 基准测试主脚本
├── example.py                # 基本使用示例
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

## 📊 基准测试

QCIT+模型基准测试框架提供了与BERT、Standard和GPT-2等模型的性能比较。

### 运行基准测试

```bash
# 运行所有基准测试
python run_benchmarks.py --epochs 3 --batch_size 16

# 仅运行情感分析任务（SST-2数据集）
python run_benchmarks.py --tasks sst2 --epochs 3

# 仅运行语言建模任务（WikiText数据集）
python run_benchmarks.py --tasks wikitext --epochs 3

# 使用GPU加速（如可用）
python run_benchmarks.py --gpu
```

### 基准测试输出

测试框架将生成以下输出：

1. **指标文件**：JSON格式的性能指标
2. **比较表格**：CSV格式的模型比较数据
3. **可视化图表**：训练曲线和性能对比图
4. **PDF报告**：完整的基准测试分析报告

所有输出文件默认保存在 `benchmark_results` 目录中。

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