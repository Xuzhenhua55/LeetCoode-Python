"""
Transformer 使用示例
展示如何使用基础的 Transformer 模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformer import Transformer


def simple_example():
    """
    简单的 Transformer 使用示例
    """
    print("=" * 60)
    print("Transformer 基础使用示例")
    print("=" * 60)

    # 设置随机种子
    torch.manual_seed(42)

    # 模型参数
    src_vocab_size = 1000  # 源语言词汇表大小
    tgt_vocab_size = 1000  # 目标语言词汇表大小
    d_model = 512  # 模型维度
    n_layers = 6  # 编码器/解码器层数
    n_heads = 8  # 注意力头数
    d_ff = 2048  # 前馈网络维度
    dropout = 0.1  # Dropout 概率
    max_len = 5000  # 最大序列长度

    # 创建模型
    print("\n1. 创建 Transformer 模型...")
    model = Transformer(src_vocab_size=src_vocab_size,
                        tgt_vocab_size=tgt_vocab_size,
                        d_model=d_model,
                        n_layers=n_layers,
                        n_heads=n_heads,
                        d_ff=d_ff,
                        dropout=dropout,
                        max_len=max_len)

    print(f"   - 源语言词汇表大小: {src_vocab_size}")
    print(f"   - 目标语言词汇表大小: {tgt_vocab_size}")
    print(f"   - 模型维度: {d_model}")
    print(f"   - 编码器/解码器层数: {n_layers}")
    print(f"   - 注意力头数: {n_heads}")
    print(f"   - 前馈网络维度: {d_ff}")

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    print(f"\n   总参数量: {total_params:,}")
    print(f"   可训练参数量: {trainable_params:,}")

    # 创建示例数据
    print("\n2. 创建示例数据...")
    batch_size = 2
    src_len = 10
    tgt_len = 8

    # 随机生成源序列和目标序列
    src = torch.randint(0, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))

    print(f"   - 批次大小: {batch_size}")
    print(f"   - 源序列长度: {src_len}")
    print(f"   - 目标序列长度: {tgt_len}")
    print(f"   - 源序列形状: {src.shape}")
    print(f"   - 目标序列形状: {tgt.shape}")

    # 前向传播
    print("\n3. 执行前向传播...")
    model.eval()
    with torch.no_grad():
        output = model(src, tgt)

    print(f"   - 输出形状: {output.shape}")
    print(
        f"   - 期望形状: [batch_size={batch_size}, tgt_len={tgt_len}, tgt_vocab_size={tgt_vocab_size}]"
    )

    # 计算损失（示例）
    print("\n4. 计算损失（示例）...")
    criterion = nn.CrossEntropyLoss()

    # 生成随机目标标签
    target_labels = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))

    # 重塑输出以计算损失
    output_reshaped = output.view(-1, tgt_vocab_size)
    target_reshaped = target_labels.view(-1)

    loss = criterion(output_reshaped, target_reshaped)
    print(f"   - 损失值: {loss.item():.4f}")

    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


def encoder_decoder_example():
    """
    分别使用编码器和解码器的示例
    """
    print("\n" + "=" * 60)
    print("编码器-解码器分离使用示例")
    print("=" * 60)

    # 创建模型
    model = Transformer(src_vocab_size=1000,
                        tgt_vocab_size=1000,
                        d_model=512,
                        n_layers=6,
                        n_heads=8,
                        d_ff=2048,
                        dropout=0.1)

    model.eval()

    # 示例数据
    batch_size = 2
    src_len = 10
    tgt_len = 8

    src = torch.randint(0, 1000, (batch_size, src_len))
    tgt = torch.randint(0, 1000, (batch_size, tgt_len))

    print("\n1. 单独使用编码器...")
    with torch.no_grad():
        encoder_output = model.encode(src)
    print(f"   - 编码器输出形状: {encoder_output.shape}")

    print("\n2. 使用编码器输出进行解码...")
    with torch.no_grad():
        decoder_output = model.decode(tgt, encoder_output)
    print(f"   - 解码器输出形状: {decoder_output.shape}")

    print("\n" + "=" * 60)


def mask_example():
    """
    掩码使用示例
    """
    print("\n" + "=" * 60)
    print("掩码使用示例")
    print("=" * 60)

    # 创建模型
    model = Transformer(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        d_model=512,
        n_layers=2,  # 使用较少的层以便快速演示
        n_heads=8,
        d_ff=2048,
        dropout=0.1)

    print("\n1. 生成目标序列掩码（防止看到未来信息）...")
    tgt_len = 5
    tgt_mask = model.generate_square_subsequent_mask(tgt_len)
    print(f"   - 掩码形状: {tgt_mask.shape}")
    print(f"   - 掩码矩阵:\n{tgt_mask}")
    print("   - 说明: 0 表示可见，-inf 表示被掩盖")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # 运行所有示例
    simple_example()
    encoder_decoder_example()
    mask_example()

    print("\n所有示例运行完成！")
    print("\n提示：这是一个基础的 Transformer 实现，适用于学习和理解。")
    print("在实际应用中，可能需要添加更多功能，如：")
    print("  - Beam Search 解码")
    print("  - 学习率调度器")
    print("  - 标签平滑")
    print("  - 模型保存和加载")
    print("  - 数据预处理和后处理")
