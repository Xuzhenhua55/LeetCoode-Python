"""
Decoder Mask 处理机制详解
展示 mask 是如何在 decoder 中逐层处理的
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def visualize_mask(mask, title="Mask"):
    """可视化掩码矩阵"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print('=' * 60)

    if torch.is_tensor(mask):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = mask

    # 打印矩阵
    for i, row in enumerate(mask_np):
        print(f"位置{i}: ", end="")
        for val in row:
            if val == 0 or val > -100:
                print("✓", end="  ")  # 可见
            else:
                print("✗", end="  ")  # 被遮挡
        print()

    print("\n数值形式:")
    print(mask_np)
    print('=' * 60)


def step1_generate_mask():
    """
    步骤1: 在 Transformer 中生成掩码
    """
    print("\n" + "=" * 80)
    print("步骤 1: 生成目标序列的前瞻掩码 (Look-ahead Mask)")
    print("=" * 80)

    seq_len = 5

    print(f"\n目标序列长度: {seq_len}")
    print("目标: 防止解码器在生成位置 i 时看到位置 i+1 及以后的信息")

    # 生成上三角矩阵
    print("\n生成上三角矩阵 (diagonal=1):")
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    print(mask)
    print("说明: 1 表示需要遮挡的位置")

    # 填充为 -inf 和 0
    print("\n将 1 替换为 -inf，0 保持为 0:")
    mask = mask.masked_fill(mask == 1, float('-inf'))
    mask = mask.masked_fill(mask == 0, float(0.0))

    visualize_mask(mask, "最终的前瞻掩码")

    print("\n解释:")
    print("  ✓ 可见: 当前位置及之前的位置")
    print("  ✗ 遮挡: 未来的位置")
    print("\n例如:")
    print("  - 位置0: 只能看到位置0 (自己)")
    print("  - 位置1: 能看到位置0,1")
    print("  - 位置2: 能看到位置0,1,2")
    print("  - 以此类推...")

    return mask


def step2_pass_to_decoder(mask):
    """
    步骤2: 将掩码传递给 Decoder
    """
    print("\n" + "=" * 80)
    print("步骤 2: 传递掩码到 Decoder")
    print("=" * 80)

    print("\n在 Transformer.forward() 中:")
    print("```python")
    print("# 通过解码器")
    print("decoder_output, _, _ = self.decoder(")
    print("    tgt_embedded,      # 目标序列嵌入")
    print("    encoder_output,    # 编码器输出")
    print("    tgt_mask,          # ← 自注意力掩码 (前瞻掩码)")
    print("    src_mask           # ← 交叉注意力掩码")
    print(")")
    print("```")

    print("\n在 Decoder.forward() 中:")
    print("```python")
    print("for layer in self.layers:")
    print("    x, self_attn, cross_attn = layer(")
    print("        x,")
    print("        encoder_output,")
    print("        self_attn_mask,   # ← 传递给每一层")
    print("        cross_attn_mask")
    print("    )")
    print("```")

    print("\n在 DecoderLayer.forward() 中:")
    print("```python")
    print("# 1. 自注意力 (使用前瞻掩码)")
    print("output, self_attn = self.self_attention(x, x, x, self_attn_mask)")
    print("                                                   ↑")
    print("                                           掩码在这里被使用")
    print("```")


def step3_adjust_mask_dimensions():
    """
    步骤3: 在多头注意力中调整掩码维度
    """
    print("\n" + "=" * 80)
    print("步骤 3: 调整掩码维度以适配多头注意力")
    print("=" * 80)

    batch_size = 2
    n_heads = 4
    seq_len = 5

    # 原始掩码
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    mask = mask.masked_fill(mask == 0, float(0.0))

    print(f"\n配置:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Number of heads: {n_heads}")
    print(f"  - Sequence length: {seq_len}")

    print(f"\n原始掩码形状: {mask.shape}")
    print("需要广播到: [batch_size, n_heads, seq_len, seq_len]")

    # 添加批次维度和头维度
    print("\n在 MultiHeadAttention.forward() 中:")
    print("```python")
    print("if mask is not None:")
    print("    mask = mask.unsqueeze(1)  # 添加头维度")
    print("```")

    mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    print(f"\n调整后的掩码形状: {mask.shape}")
    print("说明: [1, 1, seq_len, seq_len] 会自动广播到")
    print(
        f"      [batch_size={batch_size}, n_heads={n_heads}, seq_len={seq_len}, seq_len={seq_len}]"
    )

    # 验证广播
    broadcasted = mask.expand(batch_size, n_heads, seq_len, seq_len)
    print(f"\n广播后的形状: {broadcasted.shape}")
    print(f"✓ 每个批次的每个头都使用相同的掩码模式")


def step4_apply_mask():
    """
    步骤4: 在缩放点积注意力中应用掩码
    """
    print("\n" + "=" * 80)
    print("步骤 4: 在注意力计算中应用掩码")
    print("=" * 80)

    seq_len = 4
    d_k = 8

    # 模拟 Q 和 K
    torch.manual_seed(42)
    Q = torch.randn(1, 1, seq_len, d_k)
    K = torch.randn(1, 1, seq_len, d_k)

    # 计算注意力分数
    print("\n子步骤 4.1: 计算注意力分数")
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k**0.5)
    print(f"注意力分数形状: {attn_scores.shape}")
    print("注意力分数 (未应用掩码):")
    print(attn_scores[0, 0].detach().numpy())

    # 生成掩码
    print("\n子步骤 4.2: 生成掩码")
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    mask = mask.masked_fill(mask == 0, float(0.0))
    visualize_mask(mask, "掩码矩阵")

    # 应用掩码
    print("\n子步骤 4.3: 应用掩码")
    print("```python")
    print("attn = attn.masked_fill(mask == 0, -1e9)")
    print("```")

    # 注意：这里的逻辑是 mask==0 的地方填充 -1e9
    # 但我们的 mask 已经是 0 和 -inf 了
    # 所以实际应该是: attn = attn + mask (直接加)
    # 或者用不同的掩码格式

    # 为了演示，我们用正确的方式
    mask_bool = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    attn_masked = attn_scores.clone()
    attn_masked[0, 0].masked_fill_(mask_bool, -1e9)

    print("注意力分数 (应用掩码后):")
    print(attn_masked[0, 0].detach().numpy())
    print("\n说明: 未来位置被设置为 -1e9 (接近负无穷)")

    # Softmax
    print("\n子步骤 4.4: 应用 Softmax")
    attn_weights = F.softmax(attn_masked, dim=-1)
    print("注意力权重 (Softmax 后):")
    print(attn_weights[0, 0].detach().numpy())
    print("\n说明: -1e9 经过 softmax 后变成了接近 0")

    # 可视化
    print("\n可视化注意力权重:")
    visualize_attention_weights(attn_weights[0, 0])


def visualize_attention_weights(weights):
    """可视化注意力权重"""
    print("\n" + "-" * 50)
    weights_np = weights.detach().cpu().numpy()
    seq_len = weights_np.shape[0]

    print("     ", end="")
    for j in range(seq_len):
        print(f"位置{j}", end="  ")
    print()

    for i in range(seq_len):
        print(f"位置{i}", end="  ")
        for j in range(seq_len):
            w = weights_np[i, j]
            if w > 0.3:
                symbol = "█"
            elif w > 0.2:
                symbol = "▓"
            elif w > 0.1:
                symbol = "▒"
            elif w > 0.01:
                symbol = "░"
            else:
                symbol = "·"
            print(f" {symbol}  ", end="  ")
        print()

    print("\n说明: █ > 0.3, ▓ > 0.2, ▒ > 0.1, ░ > 0.01, · < 0.01")
    print("-" * 50)


def complete_example():
    """
    完整示例：从头到尾展示 mask 的处理
    """
    print("\n" + "=" * 80)
    print("完整示例：Decoder 中的 Mask 处理全流程")
    print("=" * 80)

    from transformer import Transformer

    # 创建模型
    model = Transformer(src_vocab_size=1000,
                        tgt_vocab_size=1000,
                        d_model=512,
                        n_layers=2,
                        n_heads=8,
                        d_ff=2048)

    # 创建输入
    batch_size = 2
    src_len = 10
    tgt_len = 8

    src = torch.randint(0, 1000, (batch_size, src_len))
    tgt = torch.randint(0, 1000, (batch_size, tgt_len))

    print(f"\n输入:")
    print(f"  源序列: {src.shape}")
    print(f"  目标序列: {tgt.shape}")

    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(src, tgt)

    print(f"\n输出: {output.shape}")

    # 展示自动生成的掩码
    print("\n模型自动生成的掩码:")
    tgt_mask = model.generate_square_subsequent_mask(tgt_len)
    visualize_mask(tgt_mask[:5, :5], f"前瞻掩码 (前5x5)")

    print("\n✓ 掩码已经在内部自动应用到:")
    print("  1. Decoder 的自注意力层")
    print("  2. 防止每个位置看到未来的信息")
    print("  3. 确保自回归生成的正确性")


def two_types_of_masks():
    """
    Decoder 中的两种掩码
    """
    print("\n" + "=" * 80)
    print("Decoder 中的两种掩码")
    print("=" * 80)

    print("\n1️⃣  自注意力掩码 (Self-Attention Mask)")
    print("   - 又称: 前瞻掩码 (Look-ahead Mask)")
    print("   - 目的: 防止看到未来信息")
    print("   - 形状: [tgt_len, tgt_len]")
    print("   - 应用于: DecoderLayer 的第一个注意力层")

    seq_len = 5
    self_attn_mask = torch.triu(torch.ones(seq_len, seq_len),
                                diagonal=1).bool()

    print("\n   示例 (序列长度=5):")
    print("        位置0  位置1  位置2  位置3  位置4")
    for i in range(seq_len):
        print(f"   位置{i}", end="  ")
        for j in range(seq_len):
            if self_attn_mask[i, j]:
                print("  ✗  ", end="")  # 被遮挡
            else:
                print("  ✓  ", end="")  # 可见
        print()

    print("\n2️⃣  交叉注意力掩码 (Cross-Attention Mask)")
    print("   - 又称: 源序列掩码 (Source Mask)")
    print("   - 目的: 遮挡源序列中的填充位置 (padding)")
    print("   - 形状: [src_len, src_len] 或 padding mask")
    print("   - 应用于: DecoderLayer 的第二个注意力层")

    print("\n   示例 (源序列有padding):")
    print("   源序列: ['I', 'love', 'AI', '<PAD>', '<PAD>']")
    print("   掩码:    [ ✓,   ✓,    ✓,     ✗,      ✗  ]")
    print("            可见  可见   可见   遮挡    遮挡")

    print("\n对比:")
    print("┌─────────────────────┬───────────────────┬─────────────────────┐")
    print("│       特性          │   自注意力掩码    │   交叉注意力掩码    │")
    print("├─────────────────────┼───────────────────┼─────────────────────┤")
    print("│ 目的                │ 防止看到未来      │ 遮挡填充位置        │")
    print("│ 形状                │ 下三角矩阵        │ 布尔向量/矩阵       │")
    print("│ 必需性              │ 必需              │ 可选 (有padding时)  │")
    print("│ 在训练时            │ 始终使用          │ 根据需要使用        │")
    print("│ 在推理时            │ 始终使用          │ 根据需要使用        │")
    print("└─────────────────────┴───────────────────┴─────────────────────┘")


if __name__ == "__main__":
    # 运行所有步骤

    # 步骤 1: 生成掩码
    mask = step1_generate_mask()

    # 步骤 2: 传递掩码
    step2_pass_to_decoder(mask)

    # 步骤 3: 调整维度
    step3_adjust_mask_dimensions()

    # 步骤 4: 应用掩码
    step4_apply_mask()

    # 完整示例
    complete_example()

    # 两种掩码
    two_types_of_masks()

    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print("""
Decoder 中 Mask 的完整处理流程:

1. 生成 (Transformer.generate_square_subsequent_mask)
   └─> 创建下三角掩码矩阵
   
2. 传递 (Transformer.forward -> Decoder.forward -> DecoderLayer.forward)
   └─> 逐层传递掩码参数
   
3. 调整 (MultiHeadAttention.forward)
   └─> 添加维度以适配多头
   
4. 应用 (ScaledDotProductAttention.forward)
   └─> 在注意力分数上应用掩码
   └─> masked_fill: 将未来位置设置为 -1e9
   └─> Softmax 后变成接近 0

关键代码位置:
- transformer.py:98-100  : 生成掩码
- decoder.py:54          : 使用掩码 (自注意力)
- attention.py:120-122   : 调整掩码维度
- attention.py:42-44     : 应用掩码

核心原理:
- 将未来位置的注意力分数设置为极小值 (-1e9)
- 经过 Softmax 后，这些位置的权重接近 0
- 从而实现"看不到未来信息"的效果
""")
