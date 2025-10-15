"""
解码器模块
实现 Transformer 的 Decoder 层和 Decoder 堆栈
"""

import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .feedforward import PositionWiseFeedForward


class DecoderLayer(nn.Module):
    """
    单个解码器层
    包含：自注意力机制 + 交叉注意力机制 + 前馈神经网络
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数量
            d_ff: 前馈网络隐藏层维度
            dropout: Dropout 概率
        """
        super(DecoderLayer, self).__init__()

        # 自注意力机制（带掩码）
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)

        # 交叉注意力机制（Encoder-Decoder Attention）
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)

        # 前馈神经网络
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self,
                x,
                encoder_output,
                self_attn_mask=None,
                cross_attn_mask=None):
        """
        Args:
            x: 解码器输入 [batch_size, seq_len, d_model]
            encoder_output: 编码器输出 [batch_size, src_len, d_model]
            self_attn_mask: 自注意力掩码（用于掩盖未来信息）
            cross_attn_mask: 交叉注意力掩码
        Returns:
            output: 解码器层输出 [batch_size, seq_len, d_model]
            self_attn: 自注意力权重
            cross_attn: 交叉注意力权重
        """
        # 1. 自注意力（Q、K、V 都来自解码器输入）
        output, self_attn = self.self_attention(x, x, x, self_attn_mask)

        # 2. 交叉注意力（Q 来自解码器，K、V 来自编码器）
        output, cross_attn = self.cross_attention(output, encoder_output,
                                                  encoder_output,
                                                  cross_attn_mask)

        # 3. 前馈神经网络
        output = self.feed_forward(output)

        return output, self_attn, cross_attn


class Decoder(nn.Module):
    """
    解码器堆栈
    由多个解码器层堆叠而成
    """

    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=0.1):
        """
        Args:
            n_layers: 解码器层的数量
            d_model: 模型维度
            n_heads: 注意力头数量
            d_ff: 前馈网络隐藏层维度
            dropout: Dropout 概率
        """
        super(Decoder, self).__init__()

        # 创建 n_layers 个解码器层
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self,
                x,
                encoder_output,
                self_attn_mask=None,
                cross_attn_mask=None):
        """
        Args:
            x: 解码器输入 [batch_size, tgt_len, d_model]
            encoder_output: 编码器输出 [batch_size, src_len, d_model]
            self_attn_mask: 自注意力掩码
            cross_attn_mask: 交叉注意力掩码
        Returns:
            output: 解码器输出 [batch_size, tgt_len, d_model]
            self_attn_weights: 各层的自注意力权重列表
            cross_attn_weights: 各层的交叉注意力权重列表
        """
        self_attn_weights = []
        cross_attn_weights = []

        # 依次通过每个解码器层
        for layer in self.layers:
            x, self_attn, cross_attn = layer(x, encoder_output, self_attn_mask,
                                             cross_attn_mask)
            self_attn_weights.append(self_attn)
            cross_attn_weights.append(cross_attn)

        return x, self_attn_weights, cross_attn_weights
