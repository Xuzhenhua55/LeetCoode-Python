import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class RotaryPositionalEncoding(nn.Module):
    """
    RoPE (Rotary Positional Embedding)
    通过复数旋转方式注入位置信息
    """

    def __init__(self, d_model, base=10000):
        """
        Args:
            d_model: 模型维度（需为偶数）
            base: 频率基数
        """
        super(RotaryPositionalEncoding, self).__init__()
        # 计算旋转频率: 1 / (base^(2i/d_model))
        inv_freq = 1.0 / (base
                          **(torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        
        形状变化示例 (假设 batch_size=2, seq_len=4, d_model=8):
            输入 x: [2, 4, 8]
            position: [4, 1]
            angles: [4, 4] = [seq_len, d_model/2]
            cos/sin: [1, 4, 4] = [1, seq_len, d_model/2]
            x1/x2: [2, 4, 4] = [batch_size, seq_len, d_model/2] (拆分奇偶维度)
            stack后: [2, 4, 4, 2] (最后一维是实部/虚部对)
            flatten后: [2, 4, 8] (恢复原始维度)
        """
        seq_len = x.size(1)

        # 计算位置角度: [seq_len, 1] * [1, d_model/2] -> [seq_len, d_model/2]
        position = torch.arange(seq_len).float().unsqueeze(1)
        angles = position * self.inv_freq.unsqueeze(0)  # [seq_len, d_model/2]

        # 广播到batch维度: [1, seq_len, d_model/2]
        cos = angles.cos().unsqueeze(0)  # [1, seq_len, d_model/2]
        sin = angles.sin().unsqueeze(0)

        # 拆分为偶数维度(实部)和奇数维度(虚部): [batch, seq_len, d_model/2]
        x1 = x[..., ::2]  # 实部: 索引 0,2,4,6...
        x2 = x[..., 1::2]  # 虚部: 索引 1,3,5,7...

        # 复数旋转变换: z' = (x1 + ix2) * (cos + i*sin)
        # 实部': x1*cos - x2*sin, 虚部': x1*sin + x2*cos
        x_rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos],
                                dim=-1)
        # [batch, seq_len, d_model/2, 2] -> [batch, seq_len, d_model]
        x_rotated = x_rotated.flatten(-2)

        return x_rotated
