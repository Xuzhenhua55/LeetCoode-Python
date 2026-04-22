import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        """
        d_model: 模型总维度
        n_head:  注意力头数，要求 d_model % n_head == 0
        d_k:     每个头的维度，d_k = d_model // n_head
        """
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_model = d_model
        self.n_head  = n_head
        self.d_k     = d_model // n_head   # 每个头的维度（论文中的 d_k）

        # 三个投影矩阵：将输入映射到 Q / K / V 空间
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        # 输出投影：将多头拼接结果映射回 d_model
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x:    [batch_size, seq_len, d_model]  输入序列
        mask: [batch_size, 1, seq_len, seq_len]     注意力掩码，0 的位置被屏蔽（如 padding mask / causal mask）
        return: [batch_size, seq_len, d_model]

        公式推导：
            Attention(Q, K, V) = softmax( Q @ K^T / sqrt(d_k) ) @ V
            MultiHead(x) = Concat(head_1, ..., head_h) @ W_o
            其中 head_i = Attention(x @ W_q_i, x @ W_k_i, x @ W_v_i)
        """
        batch_size, seq_len, _ = x.shape

        # Step 1: 线性投影 + 拆分多头
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, n_head, d_k] -> [batch_size, n_head, seq_len, d_k]
        Q = self.w_q(x).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)  # [batch_size, n_head, seq_len, d_k]
        K = self.w_k(x).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)  # [batch_size, n_head, seq_len, d_k]
        V = self.w_v(x).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)  # [batch_size, n_head, seq_len, d_k]

        # Step 2: 缩放点积注意力分数
        # 除以 sqrt(d_k) 防止点积值过大导致 softmax 梯度消失
        scale = math.sqrt(self.d_k)
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / scale            # [batch_size, n_head, seq_len, seq_len]

        # Step 3: 应用掩码（可选），将被屏蔽位置的分数置为 -inf，softmax 后趋近于 0
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))   # [batch_size, n_head, seq_len, seq_len]

        # Step 4: softmax 归一化，得到注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)                         # [batch_size, n_head, seq_len, seq_len]

        # Step 5: 加权求和 V，得到每个头的输出
        attn_output = torch.matmul(attn_weights, V)                           # [batch_size, n_head, seq_len, d_k]

        # Step 6: 合并多头 + 输出投影
        # [batch_size, n_head, seq_len, d_k] -> [batch_size, seq_len, n_head, d_k] -> [batch_size, seq_len, d_model]
        # contiguous(): transpose 后内存不连续，view 要求内存连续，需先调用 contiguous() 重新整理内存布局
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_output = self.w_o(attn_output)                                   # [batch_size, seq_len, d_model]

        return attn_output


# ── 验证 ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    torch.manual_seed(0)
    batch_size, seq_len, d_model, n_head = 2, 8, 64, 8

    mha = MultiHeadAttention(d_model=d_model, n_head=n_head)
    x   = torch.randn(batch_size, seq_len, d_model)

    # 无 mask
    out = mha(x)
    print(f"output shape (no mask) : {out.shape}")   # [2, 8, 64]

    # causal mask（下三角为 1，上三角为 0）
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)     # [1, 1, seq_len, seq_len]
    out_masked  = mha(x, mask=causal_mask)
    print(f"output shape (causal)  : {out_masked.shape}")                    # [2, 8, 64]