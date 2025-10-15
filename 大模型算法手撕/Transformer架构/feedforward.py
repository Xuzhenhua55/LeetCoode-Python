"""
前馈神经网络模块
实现 Position-wise Feed-Forward Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionWiseFeedForward(nn.Module):
    """
    位置前馈神经网络
    由两个线性变换组成，中间使用 ReLU 激活函数
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            d_ff: 前馈网络的隐藏层维度（通常是 d_model 的 4 倍）
            dropout: Dropout 概率
        """
        super(PositionWiseFeedForward, self).__init__()

        # 两层全连接网络
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
        Returns:
            output: 前馈网络输出 [batch_size, seq_len, d_model]
        """
        # 保存残差连接
        residual = x

        # 第一层线性变换 + ReLU
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # 第二层线性变换
        x = self.fc2(x)
        x = self.dropout(x)

        # 残差连接和层归一化
        output = self.layer_norm(x + residual)

        return output
