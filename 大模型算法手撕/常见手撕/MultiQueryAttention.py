import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        """
        d_model: 模型总维度
        n_head:  注意力头数，要求 d_model % n_head == 0
        d_k:     每个头的维度，d_k = d_model // n_head

        MQA 核心思想：
            - Q 仍然保留 n_head 个头（每个头独立）
            - K / V 只有 1 个头（所有 Q 头共享同一组 K / V）
            - 相比 MHA，K / V 的参数量和显存占用大幅减少，推理速度更快
        """
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_model = d_model
        self.n_head  = n_head
        self.d_k     = d_model // n_head   # 每个头的维度

        # Q 投影：映射到完整的多头空间 [d_model -> d_model]
        self.w_q = nn.Linear(d_model, d_model)
        # K / V 投影：只映射到单头空间 [d_model -> d_k]，所有 Q 头共享
        self.w_k = nn.Linear(d_model, self.d_k)
        self.w_v = nn.Linear(d_model, self.d_k)
        # 输出投影：将多头拼接结果映射回 d_model
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x:    [batch_size, seq_len, d_model]            输入序列
        mask: [batch_size, 1, seq_len, seq_len]         注意力掩码，0 的位置被屏蔽
        return: [batch_size, seq_len, d_model]

        公式推导：
            MQA 与 MHA 的区别仅在于 K / V 只有 1 个头：
            head_i = Attention(x @ W_q_i, x @ W_k, x @ W_v)   ← K / V 共享
            MultiQuery(x) = Concat(head_1, ..., head_h) @ W_o
        """
        batch_size, seq_len, _ = x.shape

        # Step 1: 线性投影 + 拆分多头
        # Q: [batch_size, seq_len, d_model] -> [batch_size, n_head, seq_len, d_k]
        Q = self.w_q(x).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)  # [batch_size, n_head, seq_len, d_k]
        # K / V: [batch_size, seq_len, d_k] -> [batch_size, 1, seq_len, d_k]，只有 1 个头
        K = self.w_k(x).unsqueeze(1)   # [batch_size, 1, seq_len, d_k]
        V = self.w_v(x).unsqueeze(1)   # [batch_size, 1, seq_len, d_k]

        # Step 2: 缩放点积注意力分数
        # K 会通过广播机制自动扩展到 n_head 个头参与计算
        scale = math.sqrt(self.d_k)
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / scale            # [batch_size, n_head, seq_len, seq_len]

        # Step 3: 应用掩码（可选），将被屏蔽位置的分数置为 -inf，softmax 后趋近于 0
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))   # [batch_size, n_head, seq_len, seq_len]

        # Step 4: softmax 归一化，得到注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)                         # [batch_size, n_head, seq_len, seq_len]

        # Step 5: 加权求和 V，V 同样通过广播扩展到 n_head 个头
        attn_output = torch.matmul(attn_weights, V)                           # [batch_size, n_head, seq_len, d_k]

        # Step 6: 合并多头 + 输出投影
        # [batch_size, n_head, seq_len, d_k] -> [batch_size, seq_len, n_head, d_k] -> [batch_size, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_output = self.w_o(attn_output)                                   # [batch_size, seq_len, d_model]

        return attn_output


# ── 验证 ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    torch.manual_seed(0)
    batch_size, seq_len, d_model, n_head = 2, 8, 64, 8

    mqa = MultiQueryAttention(d_model=d_model, n_head=n_head)
    x   = torch.randn(batch_size, seq_len, d_model)

    # 无 mask
    out = mqa(x)
    print(f"output shape (no mask) : {out.shape}")   # [2, 8, 64]

    # causal mask（下三角为 1，上三角为 0）
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)     # [1, 1, seq_len, seq_len]
    out_masked  = mqa(x, mask=causal_mask)
    print(f"output shape (causal)  : {out_masked.shape}")                    # [2, 8, 64]

    # 对比 MQA 与 MHA 的参数量差异
    from MultiHeadAttention import MultiHeadAttention
    mha = MultiHeadAttention(d_model=d_model, n_head=n_head)
    mqa_params = sum(p.numel() for p in mqa.parameters())
    mha_params = sum(p.numel() for p in mha.parameters())
    print(f"MHA params: {mha_params}, MQA params: {mqa_params}")             # MQA 的 K/V 参数量更少

"""
补充说明：广播机制（Broadcasting）在 MQA 中的应用

在 Step 2 中计算注意力分数时：
attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / scale

参与计算的张量形状：
- Q 的形状: [batch_size, n_head, seq_len, d_k]
- K.transpose(-1, -2) 的形状: [batch_size, 1, d_k, seq_len]

为什么形状不同还能相乘？
这里利用了 PyTorch 的广播机制。当 torch.matmul 处理多维张量时，它会将最后两个维度视为矩阵，前面的维度视为 Batch。
对比 Batch 维度：
- Q: [batch_size, n_head]
- K^T: [batch_size, 1]

根据广播规则，大小为 1 的维度会被自动"虚拟扩展"到与另一个张量相同的大小。
因此，K^T 在底层计算时会被隐式地当作 [batch_size, n_head, d_k, seq_len] 参与计算。

意义：
相当于有 n_head 份一模一样的 K^T 分别与 n_head 个不同的 Q 头进行标准的矩阵乘法。
广播机制的精妙之处在于它是"虚拟"扩展的，物理内存中始终只保存了 1 份 K。
这就是 MQA 能够大幅降低显存占用（尤其是推理时的 KV Cache）并提升生成速度的底层代码实现原理。
"""
