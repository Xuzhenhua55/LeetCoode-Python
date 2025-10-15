"""
Transformer 主模型
整合编码器、解码器和位置编码，实现完整的 Transformer 架构
"""

import torch
import torch.nn as nn
import numpy as np
from .encoder import Encoder
from .decoder import Decoder
from .positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    """
    完整的 Transformer 模型
    用于序列到序列（Seq2Seq）任务
    """

    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model=512,
                 n_layers=6,
                 n_heads=8,
                 d_ff=2048,
                 dropout=0.1,
                 max_len=5000):
        """
        Args:
            src_vocab_size: 源语言词汇表大小
            tgt_vocab_size: 目标语言词汇表大小
            d_model: 模型维度
            n_layers: 编码器/解码器层数
            n_heads: 注意力头数量
            d_ff: 前馈网络隐藏层维度
            dropout: Dropout 概率
            max_len: 序列最大长度
        """
        super(Transformer, self).__init__()

        self.d_model = d_model

        # 源序列嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)

        # 目标序列嵌入层
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_len,
                                                      dropout)

        # 编码器
        self.encoder = Encoder(n_layers, d_model, n_heads, d_ff, dropout)

        # 解码器
        self.decoder = Decoder(n_layers, d_model, n_heads, d_ff, dropout)

        # 输出层
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)

        # 初始化参数
        self._init_parameters()

    def _init_parameters(self):
        """
        参数初始化
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: 源序列 [batch_size, src_len]
            tgt: 目标序列 [batch_size, tgt_len]
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
        Returns:
            output: 模型输出 [batch_size, tgt_len, tgt_vocab_size]
        """
        # 1. 编码器部分
        # 嵌入 + 位置编码
        src_embedded = self.src_embedding(src) * np.sqrt(self.d_model)
        src_embedded = self.positional_encoding(src_embedded)

        # 通过编码器
        encoder_output, _ = self.encoder(src_embedded, src_mask)

        # 2. 解码器部分
        # 嵌入 + 位置编码
        tgt_embedded = self.tgt_embedding(tgt) * np.sqrt(self.d_model)
        tgt_embedded = self.positional_encoding(tgt_embedded)

        # 生成目标序列的自注意力掩码（防止看到未来信息）
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(
                tgt.device)

        # 通过解码器
        decoder_output, _, _ = self.decoder(tgt_embedded, encoder_output,
                                            tgt_mask, src_mask)

        # 3. 输出层
        output = self.output_linear(decoder_output)

        return output

    @staticmethod
    def generate_square_subsequent_mask(size):
        """
        生成用于掩盖未来信息的上三角掩码矩阵
        Args:
            size: 序列长度
        Returns:
            mask: 掩码矩阵 [size, size]
        """
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        mask = mask.masked_fill(mask == 0, float(0.0))
        return mask

    def encode(self, src, src_mask=None):
        """
        仅执行编码
        Args:
            src: 源序列 [batch_size, src_len]
            src_mask: 源序列掩码
        Returns:
            encoder_output: 编码器输出 [batch_size, src_len, d_model]
        """
        src_embedded = self.src_embedding(src) * np.sqrt(self.d_model)
        src_embedded = self.positional_encoding(src_embedded)
        encoder_output, _ = self.encoder(src_embedded, src_mask)
        return encoder_output

    def decode(self, tgt, encoder_output, tgt_mask=None, src_mask=None):
        """
        仅执行解码
        Args:
            tgt: 目标序列 [batch_size, tgt_len]
            encoder_output: 编码器输出 [batch_size, src_len, d_model]
            tgt_mask: 目标序列掩码
            src_mask: 源序列掩码
        Returns:
            output: 解码器输出 [batch_size, tgt_len, tgt_vocab_size]
        """
        tgt_embedded = self.tgt_embedding(tgt) * np.sqrt(self.d_model)
        tgt_embedded = self.positional_encoding(tgt_embedded)

        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(
                tgt.device)

        decoder_output, _, _ = self.decoder(tgt_embedded, encoder_output,
                                            tgt_mask, src_mask)
        output = self.output_linear(decoder_output)
        return output
