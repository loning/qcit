import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from einops import rearrange
import numpy as np

class ClassicStateInitializer(nn.Module):
    """经典态初始化模块（BERT）: 𝓑_init: 𝓣_text → 𝓒"""
    
    def __init__(self, pretrained_model="bert-base-chinese", output_dim=768):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.output_dim = output_dim
        
        if output_dim != self.bert.config.hidden_size:
            self.projection = nn.Linear(self.bert.config.hidden_size, output_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, text_input_ids, attention_mask=None):
        """
        映射: x → 𝓑𝓔𝓡𝓣(x) = ψ_c^0
        """
        outputs = self.bert(input_ids=text_input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0]  # 取[CLS]标记作为句子表示
        psi_c_0 = self.projection(cls_token)  # 初始经典表示
        return psi_c_0

class ModalBridgeEncoder(nn.Module):
    """模态桥接编码器 H: 𝕏 → 𝓒 ∪ 𝓠"""
    
    def __init__(self, dim=768, img_dim=2048):
        super().__init__()
        self.dim = dim
        self.text_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.text_encoder = ClassicStateInitializer()
        self.img_encoder = nn.Sequential(
            nn.Linear(img_dim, dim*2),
            nn.ReLU(),
            nn.Linear(dim*2, dim)
        )
    
    def forward(self, x, modality='text'):
        """
        统一模态编码函数:
        H: 𝕏 → 𝓒 ∪ 𝓠
        """
        if modality == 'text':
            if isinstance(x, str):
                inputs = self.text_tokenizer(x, return_tensors="pt")
                return self.text_encoder(inputs.input_ids, inputs.attention_mask)
            else:
                return self.text_encoder(x['input_ids'], x['attention_mask'])
        elif modality == 'image':
            return self.img_encoder(x)
        elif modality == 'numeric':
            return torch.sign(x)  # sign(x) for numeric data
        else:
            raise ValueError(f"Unsupported modality: {modality}")

class ExtendedMRAO(nn.Module):
    """扩展MRAO轨迹算子: MRAO*"""
    
    def __init__(self, dim=768, window_size=5, gamma=0.85):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.gamma = gamma
        self.register_buffer('gammas', torch.tensor([gamma**k for k in range(1, window_size+1)]))
    
    def forward(self, current_state, trajectory_history):
        """
        窗口化演化公式:
        MRAO*(Ψ_t) = Ψ_t + ∑_{k=1}^{W} γ^k · (Ψ_{t-k} - Ψ_{t-k-1})
        """
        batch_size = current_state.shape[0]
        
        # 确保历史轨迹长度足够
        if len(trajectory_history) < self.window_size + 1:
            # 如果历史不够长，只使用可用的历史
            available_window = len(trajectory_history) - 1
            if available_window <= 0:
                return current_state
            
            # 只使用可用的轨迹
            gammas = self.gammas[:available_window]
            window_sum = torch.zeros_like(current_state)
            
            for k in range(available_window):
                if k+1 < len(trajectory_history):
                    diff = trajectory_history[-(k+1)] - trajectory_history[-(k+2)]
                    window_sum += gammas[k] * diff
            
            return current_state + window_sum
        
        # 完整窗口
        window_sum = torch.zeros_like(current_state)
        for k in range(self.window_size):
            diff = trajectory_history[-(k+1)] - trajectory_history[-(k+2)]
            window_sum += self.gammas[k] * diff
        
        return current_state + window_sum

class StructurePreferenceScorer(nn.Module):
    """结构偏好打分函数: 𝓛_pref"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, trajectory):
        """
        定义:
        𝓛_pref(𝓣) = (1/|𝓣|) ∑_t Var(Ψ_t)
        评估结构轨迹在演化过程中的熵激活密度与稳定性
        """
        if not trajectory:
            return torch.tensor(0.0)
        
        variances = [torch.var(state) for state in trajectory]
        mean_variance = sum(variances) / len(variances)
        return mean_variance

class ControlTensorGenerator(nn.Module):
    """控制张量生成器: 𝓣_ctrl"""
    
    def __init__(self, control_dim=64, output_dim=768):
        super().__init__()
        self.mapping = nn.Linear(control_dim, output_dim)
    
    def forward(self, z):
        """
        定义控制信号空间:
        z ∈ ℝ^k, 𝓣_ctrl(z) = σ(Wz) ∈ 𝔾
        """
        return torch.sigmoid(self.mapping(z))

class SRCO(nn.Module):
    """结构反演计算算子: SRCO"""
    
    def __init__(self, dim=768):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads=8)
        self.norm1 = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        # 自注意力机制
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

class QCDAO(nn.Module):
    """量子经典双重算子: QCDAO"""
    
    def __init__(self, dim=768):
        super().__init__()
        self.quantum_proj = nn.Linear(dim, dim)
        self.classical_proj = nn.Linear(dim, dim)
        self.merge = nn.Linear(dim*2, dim)
    
    def forward(self, state):
        quantum_component = self.quantum_proj(state)
        classical_component = self.classical_proj(state)
        
        # 量子叠加
        quantum_component = F.gelu(quantum_component)
        
        # 经典投影
        classical_component = torch.tanh(classical_component)
        
        # 合并
        merged = torch.cat([quantum_component, classical_component], dim=-1)
        return self.merge(merged)

class QCITModel(nn.Module):
    """量子经典同构变换器: QCIT+"""
    
    def __init__(self, dim=768, control_dim=64, window_size=5):
        super().__init__()
        self.dim = dim
        
        # 初始化所有子模块
        self.B_init = ClassicStateInitializer(output_dim=dim)
        self.H = ModalBridgeEncoder(dim=dim)
        self.SRCO = SRCO(dim=dim)
        self.MRAO = ExtendedMRAO(dim=dim, window_size=window_size)
        self.QCDAO = QCDAO(dim=dim)
        self.L_pref = StructurePreferenceScorer()
        self.T_ctrl = ControlTensorGenerator(control_dim=control_dim, output_dim=dim)
        
        # 轨迹历史
        self.trajectory_history = []
        
        # 映射头
        self.output_head = nn.Linear(dim, dim)
    
    def forward(self, x, modality='text', control_signal=None, initialize=True):
        """
        最终更新后的演化主律（结构反馈 + 控制）:
        Ψ_{t+1} = SRCO(Ψ_t) + MRAO*(𝓣_{0:t}) + 𝔾_t
        """
        if initialize:
            # 使用适当的初始化器
            if modality == 'text':
                state = self.B_init(x['input_ids'], x.get('attention_mask'))
            else:
                state = self.H(x, modality)
            
            # 重置轨迹历史
            self.trajectory_history = [state.detach()]
        else:
            # 使用最近的状态
            state = self.trajectory_history[-1]
        
        # 应用SRCO
        state = self.SRCO(state)
        
        # 应用扩展MRAO（考虑轨迹历史）
        if len(self.trajectory_history) > 1:
            state = self.MRAO(state, self.trajectory_history)
        
        # 应用控制张量（如果提供）
        if control_signal is not None:
            G_t = self.T_ctrl(control_signal)
            state = state + G_t
        
        # 应用QCDAO
        state = self.QCDAO(state)
        
        # 更新轨迹历史
        self.trajectory_history.append(state.detach())
        
        # 如果历史过长，保持固定长度
        max_history = 50  # 合理的历史长度上限
        if len(self.trajectory_history) > max_history:
            self.trajectory_history = self.trajectory_history[-max_history:]
        
        # 计算偏好分数（仅用于监控）
        pref_score = self.L_pref(self.trajectory_history)
        
        # 输出最终状态
        output = self.output_head(state)
        
        return output, pref_score
    
    def evolve_trajectory(self, steps=10, control_signals=None):
        """
        沿着轨迹演化指定的步数
        """
        if not self.trajectory_history:
            raise ValueError("需要先初始化轨迹")
        
        outputs = []
        scores = []
        
        for i in range(steps):
            control_signal = None
            if control_signals is not None and i < len(control_signals):
                control_signal = control_signals[i]
            
            output, score = self.forward(None, initialize=False, control_signal=control_signal)
            outputs.append(output)
            scores.append(score)
        
        return outputs, scores 