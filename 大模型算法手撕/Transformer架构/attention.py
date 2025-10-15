"""
注意力机制模块
实现缩放点积注意力和多头注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """

    def __init__(self, temperature, dropout=0.1):
        """
        Args:
            temperature: 缩放因子，通常为 sqrt(d_k)
            dropout: Dropout 概率
        """
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: Query 张量 [batch_size, n_heads, seq_len_q, d_k]
            k: Key 张量 [batch_size, n_heads, seq_len_k, d_k]
            v: Value 张量 [batch_size, n_heads, seq_len_v, d_v]
            mask: 掩码张量 [batch_size, 1, seq_len_q, seq_len_k]
        Returns:
            output: 注意力输出 [batch_size, n_heads, seq_len_q, d_v]
            attn: 注意力权重 [batch_size, n_heads, seq_len_q, seq_len_k]
        """
        # 计算注意力分数: QK^T / sqrt(d_k)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.temperature

        # 应用掩码（可选）
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        # 应用 softmax 获得注意力权重
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 加权求和得到输出
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    将 Q、K、V 分成多个头，并行计算注意力
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头的数量
            dropout: Dropout 概率
        """
        super(MultiHeadAttention, self).__init__()

        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads

        # 线性投影层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # 缩放点积注意力
        self.attention = ScaledDotProductAttention(temperature=math.sqrt(
            self.d_k),
                                                   dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: Query 张量 [batch_size, seq_len_q, d_model]
            k: Key 张量 [batch_size, seq_len_k, d_model]
            v: Value 张量 [batch_size, seq_len_v, d_model]
            mask: 掩码张量
        Returns:
            output: 多头注意力输出 [batch_size, seq_len_q, d_model]
            attn: 注意力权重
        """
        batch_size, seq_len_q = q.size(0), q.size(1)
        seq_len_k, seq_len_v = k.size(1), v.size(1)

        # 保存残差连接
        residual = q

        # 线性投影并分成多头
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, n_heads, d_k]
        q = self.w_q(q).view(batch_size, seq_len_q, self.n_heads, self.d_k)
        k = self.w_k(k).view(batch_size, seq_len_k, self.n_heads, self.d_k)
        v = self.w_v(v).view(batch_size, seq_len_v, self.n_heads, self.d_v)

        # 转置以进行批量矩阵乘法
        # [batch_size, seq_len, n_heads, d_k] -> [batch_size, n_heads, seq_len, d_k]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 调整掩码维度
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len_q, seq_len_k]

        # 计算注意力
        output, attn = self.attention(q, k, v, mask=mask)

        # 合并多头
        # [batch_size, n_heads, seq_len_q, d_v] -> [batch_size, seq_len_q, n_heads, d_v]
        output = output.transpose(1, 2).contiguous()
        # [batch_size, seq_len_q, n_heads, d_v] -> [batch_size, seq_len_q, d_model]
        output = output.view(batch_size, seq_len_q, self.d_model)

        # 最终线性投影
        output = self.w_o(output)
        output = self.dropout(output)

        # 残差连接和层归一化
        output = self.layer_norm(output + residual)

        return output, attn
