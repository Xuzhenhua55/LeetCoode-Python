"""
编码器模块
实现 Transformer 的 Encoder 层和 Encoder 堆栈
"""

import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .feedforward import PositionWiseFeedForward


class EncoderLayer(nn.Module):
    """
    单个编码器层
    包含：自注意力机制 + 前馈神经网络
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数量
            d_ff: 前馈网络隐藏层维度
            dropout: Dropout 概率
        """
        super(EncoderLayer, self).__init__()

        # 自注意力机制
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)

        # 前馈神经网络
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 掩码张量（可选）
        Returns:
            output: 编码器层输出 [batch_size, seq_len, d_model]
            attn: 注意力权重
        """
        # 自注意力（Q、K、V 都来自同一个输入）
        output, attn = self.self_attention(x, x, x, mask)

        # 前馈神经网络
        output = self.feed_forward(output)

        return output, attn


class Encoder(nn.Module):
    """
    编码器堆栈
    由多个编码器层堆叠而成
    """

    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=0.1):
        """
        Args:
            n_layers: 编码器层的数量
            d_model: 模型维度
            n_heads: 注意力头数量
            d_ff: 前馈网络隐藏层维度
            dropout: Dropout 概率
        """
        super(Encoder, self).__init__()

        # 创建 n_layers 个编码器层
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, mask=None):
        """
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 掩码张量（可选）
        Returns:
            output: 编码器输出 [batch_size, seq_len, d_model]
            attn_weights: 各层的注意力权重列表
        """
        attn_weights = []

        # 依次通过每个编码器层
        for layer in self.layers:
            x, attn = layer(x, mask)
            attn_weights.append(attn)

        return x, attn_weights
