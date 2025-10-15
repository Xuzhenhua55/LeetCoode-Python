"""
位置编码模块
实现 Transformer 的位置编码机制
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    位置编码层
    使用正弦和余弦函数为序列中的每个位置生成固定的编码
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Args:
            d_model: 模型的维度
            max_len: 序列的最大长度
            dropout: Dropout 概率
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算分母项: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))

        # 偶数位置使用 sin，奇数位置使用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 添加批次维度 [max_len, d_model] -> [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # 将 pe 注册为 buffer，不作为模型参数
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
        Returns:
            添加位置编码后的张量 [batch_size, seq_len, d_model]
        """
        # 将位置编码添加到输入上
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
