# Transformer 架构实现

这是一个完整的、模块化的 Transformer 架构实现，基于论文 "Attention is All You Need"。

## 📁 项目结构

```
Transformer架构/
├── __init__.py              # 包初始化文件
├── transformer.py           # 完整的 Transformer 模型
├── encoder.py               # 编码器层和编码器堆栈
├── decoder.py               # 解码器层和解码器堆栈
├── attention.py             # 多头注意力机制
├── feedforward.py           # 位置前馈神经网络
├── positional_encoding.py   # 位置编码
├── example.py               # 使用示例
└── README.md                # 说明文档
```

## 🧩 核心组件

### 1. 位置编码 (Positional Encoding)
- **文件**: `positional_encoding.py`
- **功能**: 为序列添加位置信息
- **公式**: 
  - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
  - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

### 2. 多头注意力 (Multi-Head Attention)
- **文件**: `attention.py`
- **组件**:
  - `ScaledDotProductAttention`: 缩放点积注意力
  - `MultiHeadAttention`: 多头注意力机制
- **公式**: Attention(Q, K, V) = softmax(QK^T / √d_k)V

### 3. 前馈神经网络 (Feed-Forward Network)
- **文件**: `feedforward.py`
- **功能**: 位置独立的前馈网络
- **结构**: Linear → ReLU → Linear
- **公式**: FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

### 4. 编码器 (Encoder)
- **文件**: `encoder.py`
- **组件**:
  - `EncoderLayer`: 单个编码器层
  - `Encoder`: 编码器堆栈
- **结构**: 自注意力 → 前馈网络

### 5. 解码器 (Decoder)
- **文件**: `decoder.py`
- **组件**:
  - `DecoderLayer`: 单个解码器层
  - `Decoder`: 解码器堆栈
- **结构**: 自注意力 → 交叉注意力 → 前馈网络

### 6. Transformer 模型
- **文件**: `transformer.py`
- **功能**: 整合所有组件的完整模型
- **特点**: 
  - 支持编码器-解码器架构
  - 可分离使用编码器和解码器
  - 自动生成掩码

## 🚀 快速开始

### 安装依赖

```bash
pip install torch numpy
```

### 基础使用

```python
import torch
from transformer import Transformer

# 创建模型
model = Transformer(
    src_vocab_size=1000,    # 源语言词汇表大小
    tgt_vocab_size=1000,    # 目标语言词汇表大小
    d_model=512,            # 模型维度
    n_layers=6,             # 编码器/解码器层数
    n_heads=8,              # 注意力头数
    d_ff=2048,              # 前馈网络维度
    dropout=0.1             # Dropout 概率
)

# 准备数据
batch_size = 2
src_len = 10
tgt_len = 8

src = torch.randint(0, 1000, (batch_size, src_len))  # 源序列
tgt = torch.randint(0, 1000, (batch_size, tgt_len))  # 目标序列

# 前向传播
output = model(src, tgt)  # 输出形状: [batch_size, tgt_len, tgt_vocab_size]
```

### 运行示例

```bash
python example.py
```

## 📊 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `src_vocab_size` | - | 源语言词汇表大小 |
| `tgt_vocab_size` | - | 目标语言词汇表大小 |
| `d_model` | 512 | 模型维度 |
| `n_layers` | 6 | 编码器/解码器层数 |
| `n_heads` | 8 | 注意力头数量 |
| `d_ff` | 2048 | 前馈网络隐藏层维度 |
| `dropout` | 0.1 | Dropout 概率 |
| `max_len` | 5000 | 序列最大长度 |

## 🔧 高级用法

### 分离使用编码器和解码器

```python
# 仅使用编码器
encoder_output = model.encode(src)

# 使用编码器输出进行解码
decoder_output = model.decode(tgt, encoder_output)
```

### 使用掩码

```python
# 生成目标序列掩码（防止看到未来信息）
tgt_mask = model.generate_square_subsequent_mask(tgt_len)

# 在前向传播时使用掩码
output = model(src, tgt, src_mask=None, tgt_mask=tgt_mask)
```

## 🎯 模型特点

### ✅ 优点
- **模块化设计**: 每个组件独立实现，易于理解和修改
- **完整实现**: 包含 Transformer 的所有核心组件
- **灵活使用**: 支持完整模型和分离的编码器/解码器
- **详细注释**: 代码包含详细的中文注释
- **学习友好**: 适合学习和理解 Transformer 架构

### 📌 适用场景
- 机器翻译
- 文本摘要
- 问答系统
- 序列到序列任务
- 学习和研究

## 📚 参考资料

- **论文**: [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- **作者**: Vaswani et al., 2017

## 🔍 关键概念

### 注意力机制
注意力机制允许模型在处理序列时关注不同位置的信息。

### 多头注意力
将注意力机制分成多个头，每个头可以学习不同的表示子空间。

### 位置编码
由于 Transformer 不使用循环结构，需要显式添加位置信息。

### 残差连接和层归一化
每个子层后都使用残差连接和层归一化，有助于训练深层网络。

### 自注意力 vs 交叉注意力
- **自注意力**: Q、K、V 来自同一序列
- **交叉注意力**: Q 来自一个序列，K、V 来自另一个序列

## 🛠️ 扩展建议

如果你想进一步完善这个实现，可以考虑添加：

1. **训练循环**: 完整的训练和验证流程
2. **Beam Search**: 改进的解码策略
3. **学习率调度**: Warm-up 和衰减策略
4. **标签平滑**: 改善模型泛化
5. **模型保存/加载**: 检查点管理
6. **可视化**: 注意力权重可视化
7. **预处理**: 数据预处理和后处理
8. **评估指标**: BLEU、ROUGE 等

## 📝 注意事项

- 这是一个教学用的基础实现，未针对性能优化
- 在实际应用中可能需要根据具体任务进行调整
- 建议在 GPU 上训练大规模模型
- 需要准备合适的数据集和词汇表

## 🤝 贡献

欢迎提出问题和建议！

## 📄 许可

本项目仅供学习和研究使用。

