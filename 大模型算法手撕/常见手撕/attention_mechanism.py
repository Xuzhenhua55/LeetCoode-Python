"""
注意力机制实现
包含单头注意力和多头注意力机制的完整实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SingleHeadAttention(nn.Module):
    """
    单头自注意力机制
    最基础的注意力实现，帮助理解注意力的核心原理
    """

    def __init__(self, d_model, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            dropout: Dropout 概率
        """
        super(SingleHeadAttention, self).__init__()

        self.d_model = d_model
        self.d_k = d_model  # 单头：d_k = d_model

        # 线性投影层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # 层归一化
        self.layer_norm = nn.LayerNorm(d_model)

        # 缩放因子
        self.scale = math.sqrt(self.d_k)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: Query 张量 [batch_size, seq_len_q, d_model]
            k: Key 张量 [batch_size, seq_len_k, d_model]
            v: Value 张量 [batch_size, seq_len_v, d_model]
            mask: 掩码张量（可选）
        Returns:
            output: 注意力输出 [batch_size, seq_len_q, d_model]
            attn_weights: 注意力权重 [batch_size, seq_len_q, seq_len_k]
        """
        batch_size = q.size(0)
        seq_len_q = q.size(1)

        # 保存残差连接
        residual = q

        # 1. 线性投影
        # [batch_size, seq_len, d_model]
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)

        # 2. 计算注意力分数: QK^T / sqrt(d_k)
        # [batch_size, seq_len_q, d_model] × [batch_size, d_model, seq_len_k]
        # -> [batch_size, seq_len_q, seq_len_k]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 3. 应用掩码（如果有）
        if mask is not None:
            # 确保掩码的形状正确
            if mask.dim() == 4:
                mask = mask.squeeze(
                    1
                )  # [batch_size, 1, seq_len_q, seq_len_k] -> [batch_size, seq_len_q, seq_len_k]
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 4. Softmax 得到注意力权重
        # [batch_size, seq_len_q, seq_len_k]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 5. 加权求和
        # [batch_size, seq_len_q, seq_len_k] × [batch_size, seq_len_k, d_model]
        # -> [batch_size, seq_len_q, d_model]
        output = torch.matmul(attn_weights, V)

        # 6. 最终线性投影
        output = self.w_o(output)
        output = self.dropout(output)

        # 7. 残差连接和层归一化
        output = self.layer_norm(output + residual)

        return output, attn_weights


class SimpleSingleHeadAttention(nn.Module):
    """
    简化版单头自注意力
    去掉了残差连接和层归一化，更纯粹的注意力机制
    """

    def __init__(self, d_model, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            dropout: Dropout 概率
        """
        super(SimpleSingleHeadAttention, self).__init__()

        self.d_model = d_model
        self.d_k = d_model

        # 线性投影层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: Query 张量 [batch_size, seq_len_q, d_model]
            k: Key 张量 [batch_size, seq_len_k, d_model]
            v: Value 张量 [batch_size, seq_len_v, d_model]
            mask: 掩码张量（可选）
        Returns:
            output: 注意力输出 [batch_size, seq_len_q, d_model]
            attn_weights: 注意力权重 [batch_size, seq_len_q, seq_len_k]
        """
        # 线性投影
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)

        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 应用掩码
        if mask is not None:
            if mask.dim() == 4:
                mask = mask.squeeze(1)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        output = torch.matmul(attn_weights, V)

        return output, attn_weights


class MinimalAttention(nn.Module):
    """
    最小化注意力实现
    展示注意力机制的最核心部分，没有任何额外的线性层
    """

    def __init__(self):
        super(MinimalAttention, self).__init__()

    def forward(self, q, k, v, mask=None):
        """
        最简单的注意力计算：Attention(Q,K,V) = softmax(QK^T/√d_k)V
        
        Args:
            q: Query 张量 [batch_size, seq_len_q, d_k]
            k: Key 张量 [batch_size, seq_len_k, d_k]
            v: Value 张量 [batch_size, seq_len_v, d_v]
            mask: 掩码张量（可选）
        Returns:
            output: 注意力输出 [batch_size, seq_len_q, d_v]
            attn_weights: 注意力权重 [batch_size, seq_len_q, seq_len_k]
        """
        d_k = q.size(-1)

        # 步骤1: 计算注意力分数 QK^T
        # [batch_size, seq_len_q, d_k] × [batch_size, d_k, seq_len_k]
        # -> [batch_size, seq_len_q, seq_len_k]
        attn_scores = torch.matmul(q, k.transpose(-2, -1))

        # 步骤2: 缩放（除以 sqrt(d_k)）
        attn_scores = attn_scores / math.sqrt(d_k)

        # 步骤3: 应用掩码（可选）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 步骤4: Softmax 归一化
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 步骤5: 加权求和
        # [batch_size, seq_len_q, seq_len_k] × [batch_size, seq_len_k, d_v]
        # -> [batch_size, seq_len_q, d_v]
        output = torch.matmul(attn_weights, v)

        return output, attn_weights


# ============================================================================
# 多头注意力机制
# ============================================================================


class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    
    这是多头注意力的基础组件，每个头都会使用这个机制
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
    
    核心思想：
    1. 将 d_model 维度分成 n_heads 个子空间
    2. 每个头在自己的子空间中独立计算注意力
    3. 不同的头可以关注不同的表示子空间
    4. 最后拼接所有头的输出
    
    公式：
    MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O
    其中 head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
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
        self.d_k = d_model // n_heads  # 每个头的维度
        self.d_v = d_model // n_heads

        # 线性投影层
        # 注意：虽然有 n_heads 个头，但我们用一个大的矩阵一次性投影所有头
        self.w_q = nn.Linear(d_model, d_model)  # d_model -> n_heads * d_k
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)  # 输出投影

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

        # ============================================================
        # 步骤 1: 线性投影并分成多头
        # ============================================================
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, n_heads, d_k]
        q = q.view(batch_size, seq_len_q, self.n_heads, self.d_k)
        k = k.view(batch_size, seq_len_k, self.n_heads, self.d_k)
        v = v.view(batch_size, seq_len_v, self.n_heads, self.d_v)

        # ============================================================
        # 步骤 2: 转置以进行批量矩阵乘法
        # ============================================================
        # [batch_size, seq_len, n_heads, d_k] -> [batch_size, n_heads, seq_len, d_k]
        # 这样做是为了让每个头独立计算，形状变为 [batch*n_heads, seq_len, d_k]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ============================================================
        # 步骤 3: 调整掩码维度
        # ============================================================
        if mask is not None:
            # [batch_size, seq_len, seq_len] -> [batch_size, 1, seq_len, seq_len]
            mask = mask.unsqueeze(1)  # 为 n_heads 维度添加维度，用于广播

        # ============================================================
        # 步骤 4: 计算缩放点积注意力
        # ============================================================
        # output: [batch_size, n_heads, seq_len_q, d_v]
        # attn: [batch_size, n_heads, seq_len_q, seq_len_k]
        output, attn = self.attention(q, k, v, mask=mask)

        # ============================================================
        # 步骤 5: 合并多头
        # ============================================================
        # [batch_size, n_heads, seq_len_q, d_v] -> [batch_size, seq_len_q, n_heads, d_v]
        output = output.transpose(1, 2).contiguous()

        # [batch_size, seq_len_q, n_heads, d_v] -> [batch_size, seq_len_q, n_heads*d_v]
        # 即 [batch_size, seq_len_q, d_model]
        output = output.view(batch_size, seq_len_q, self.d_model)

        # ============================================================
        # 步骤 6: 最终线性投影
        # ============================================================
        output = self.w_o(output)
        output = self.dropout(output)

        # ============================================================
        # 步骤 7: 残差连接和层归一化
        # ============================================================
        output = self.layer_norm(output + residual)

        return output, attn


class SimpleMultiHeadAttention(nn.Module):
    """
    简化版多头注意力
    去掉残差连接和层归一化，更清晰地展示多头注意力的核心逻辑
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数量
            dropout: Dropout 概率
        """
        super(SimpleMultiHeadAttention, self).__init__()

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

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: Query 张量 [batch_size, seq_len_q, d_model]
            k: Key 张量 [batch_size, seq_len_k, d_model]
            v: Value 张量 [batch_size, seq_len_v, d_model]
            mask: 掩码张量
        Returns:
            output: 注意力输出 [batch_size, seq_len_q, d_model]
            attn: 注意力权重 [batch_size, n_heads, seq_len_q, seq_len_k]
        """
        batch_size = q.size(0)
        seq_len_q = q.size(1)

        # 线性投影
        q = self.w_q(q)  # [batch_size, seq_len_q, d_model]
        k = self.w_k(k)  # [batch_size, seq_len_k, d_model]
        v = self.w_v(v)  # [batch_size, seq_len_v, d_model]

        # 分成多头
        q = q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # 应用掩码
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Softmax
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        # 加权求和
        output = torch.matmul(attn, v)

        # 合并多头
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len_q, self.d_model)

        # 最终投影
        output = self.w_o(output)

        return output, attn
