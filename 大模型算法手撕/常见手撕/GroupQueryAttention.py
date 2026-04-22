import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, n_kv_head: int):
        """
        d_model:   模型总维度
        n_head:    Q 的注意力头数，要求 d_model % n_head == 0
        n_kv_head: K / V 的头数，要求 n_head % n_kv_head == 0
        d_k:       每个头的维度，d_k = d_model // n_head

        GQA 核心思想（MHA 与 MQA 的折中方案）：
            - Q 仍然保留 n_head 个头
            - K / V 使用 n_kv_head 个头（n_kv_head 个 K/V 头各自服务一组 Q 头）
            - 每组包含 n_head // n_kv_head 个 Q 头共享同一对 K / V
            - 当 n_kv_head == n_head 时退化为 MHA
            - 当 n_kv_head == 1     时退化为 MQA
        """
        super().__init__()
        assert d_model % n_head == 0,    "d_model must be divisible by n_head"
        assert n_head  % n_kv_head == 0, "n_head must be divisible by n_kv_head"
        self.d_model   = d_model
        self.n_head    = n_head
        self.n_kv_head = n_kv_head
        self.n_rep     = n_head // n_kv_head   # 每个 KV 头被多少个 Q 头共享
        self.d_k       = d_model // n_head     # 每个头的维度

        # Q 投影：映射到完整的多头空间 [d_model -> d_model]
        self.w_q = nn.Linear(d_model, d_model)
        # K / V 投影：映射到 n_kv_head 个头的空间 [d_model -> n_kv_head * d_k]
        self.w_k = nn.Linear(d_model, n_kv_head * self.d_k)
        self.w_v = nn.Linear(d_model, n_kv_head * self.d_k)
        # 输出投影：将多头拼接结果映射回 d_model
        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        将 K / V 的每个头沿头维度重复 n_rep 次，使其与 Q 的头数对齐。
        x:     [batch_size, n_kv_head, seq_len, d_k]
        return:[batch_size, n_head,    seq_len, d_k]   (n_head = n_kv_head * n_rep)
        """
        if n_rep == 1:
            return x
        batch_size, n_kv_head, seq_len, d_k = x.shape
        # unsqueeze + expand + reshape 实现高效重复（不复制内存，expand 是视图操作）
        return (
            x.unsqueeze(2)                                          # [batch_size, n_kv_head, 1,     seq_len, d_k]
             .expand(batch_size, n_kv_head, n_rep, seq_len, d_k)   # [batch_size, n_kv_head, n_rep, seq_len, d_k]
             .reshape(batch_size, n_kv_head * n_rep, seq_len, d_k) # [batch_size, n_head,    seq_len, d_k]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x:    [batch_size, seq_len, d_model]            输入序列
        mask: [batch_size, 1, seq_len, seq_len]         注意力掩码，0 的位置被屏蔽
        return: [batch_size, seq_len, d_model]

        公式推导：
            将 n_head 个 Q 头分成 n_kv_head 组，每组 n_rep = n_head // n_kv_head 个头：
            group_i 中的 head_j = Attention(x @ W_q_{i*n_rep+j}, x @ W_k_i, x @ W_v_i)
            GQA(x) = Concat(head_1, ..., head_h) @ W_o
        """
        batch_size, seq_len, _ = x.shape

        # Step 1: 线性投影 + 拆分多头
        # Q: [batch_size, seq_len, d_model]         -> [batch_size, n_head,    seq_len, d_k]
        Q = self.w_q(x).view(batch_size, seq_len, self.n_head,    self.d_k).transpose(1, 2)  # [batch_size, n_head,    seq_len, d_k]
        # K / V: [batch_size, seq_len, n_kv_head*d_k] -> [batch_size, n_kv_head, seq_len, d_k]
        K = self.w_k(x).view(batch_size, seq_len, self.n_kv_head, self.d_k).transpose(1, 2)  # [batch_size, n_kv_head, seq_len, d_k]
        V = self.w_v(x).view(batch_size, seq_len, self.n_kv_head, self.d_k).transpose(1, 2)  # [batch_size, n_kv_head, seq_len, d_k]

        # Step 2: 将 K / V 重复扩展，使头数与 Q 对齐（每个 KV 头重复 n_rep 次）
        K = self.repeat_kv(K, self.n_rep)   # [batch_size, n_head, seq_len, d_k]
        V = self.repeat_kv(V, self.n_rep)   # [batch_size, n_head, seq_len, d_k]

        # Step 3: 缩放点积注意力分数
        # 除以 sqrt(d_k) 防止点积值过大导致 softmax 梯度消失
        scale = math.sqrt(self.d_k)
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / scale            # [batch_size, n_head, seq_len, seq_len]

        # Step 4: 应用掩码（可选），将被屏蔽位置的分数置为 -inf，softmax 后趋近于 0
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))   # [batch_size, n_head, seq_len, seq_len]

        # Step 5: softmax 归一化，得到注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)                         # [batch_size, n_head, seq_len, seq_len]

        # Step 6: 加权求和 V，得到每个头的输出
        attn_output = torch.matmul(attn_weights, V)                           # [batch_size, n_head, seq_len, d_k]

        # Step 7: 合并多头 + 输出投影
        # [batch_size, n_head, seq_len, d_k] -> [batch_size, seq_len, n_head, d_k] -> [batch_size, seq_len, d_model]
        # contiguous(): transpose 后内存不连续，view 要求内存连续，需先调用 contiguous() 重新整理内存布局
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_output = self.w_o(attn_output)                                   # [batch_size, seq_len, d_model]

        return attn_output


# ── 验证 ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    torch.manual_seed(0)
    batch_size, seq_len, d_model, n_head = 2, 8, 64, 8

    # GQA: 8 个 Q 头，2 个 KV 头，每组 4 个 Q 头共享 1 对 KV
    gqa = GroupQueryAttention(d_model=d_model, n_head=n_head, n_kv_head=2)
    x   = torch.randn(batch_size, seq_len, d_model)

    # 无 mask
    out = gqa(x)
    print(f"output shape (no mask) : {out.shape}")   # [2, 8, 64]

    # causal mask（下三角为 1，上三角为 0）
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)     # [1, 1, seq_len, seq_len]
    out_masked  = gqa(x, mask=causal_mask)
    print(f"output shape (causal)  : {out_masked.shape}")                    # [2, 8, 64]

    # 验证退化情况：n_kv_head == n_head 时等价于 MHA，n_kv_head == 1 时等价于 MQA
    gqa_as_mha = GroupQueryAttention(d_model=d_model, n_head=n_head, n_kv_head=n_head)  # 退化为 MHA
    gqa_as_mqa = GroupQueryAttention(d_model=d_model, n_head=n_head, n_kv_head=1)       # 退化为 MQA
    print(f"GQA(n_kv=n_head) output: {gqa_as_mha(x).shape}")                # [2, 8, 64]
    print(f"GQA(n_kv=1)      output: {gqa_as_mqa(x).shape}")                # [2, 8, 64]

    # 对比三种 Attention 的参数量
    from MultiHeadAttention  import MultiHeadAttention
    from MultiQueryAttention import MultiQueryAttention
    mha = MultiHeadAttention(d_model=d_model, n_head=n_head)
    mqa = MultiQueryAttention(d_model=d_model, n_head=n_head)
    mha_params = sum(p.numel() for p in mha.parameters())
    mqa_params = sum(p.numel() for p in mqa.parameters())
    gqa_params = sum(p.numel() for p in gqa.parameters())
    print(f"MHA params: {mha_params}, GQA params: {gqa_params}, MQA params: {mqa_params}")

"""
补充说明：repeat_kv 函数中的张量操作解析 (unsqueeze -> expand -> reshape)

这段代码是 GQA 的核心操作之一：将数量较少的 K/V 头“复制”扩展，使其数量与 Q 头对齐。
假设有 6 个 Q 头 (n_head=6)，2 个 K/V 头 (n_kv_head=2)，每个 K/V 头需要被 3 个 Q 头共享 (n_rep = 3)。
初始 K/V 形状: [batch_size, n_kv_head, seq_len, d_k]，假设 n_kv_head=2，头为 [A, B]。

第一步：unsqueeze(2)
x.unsqueeze(2) -> 形状变为: [batch_size, n_kv_head, 1, seq_len, d_k]
作用：在第 2 个维度插入一个大小为 1 的新维度。原本的头 [A, B] 变成了 [[A], [B]]。

第二步：expand(...)
.expand(batch_size, n_kv_head, n_rep, seq_len, d_k) -> 形状变为: [batch_size, n_kv_head, n_rep, seq_len, d_k]
作用：将大小为 1 的维度扩展为 n_rep。[[A], [B]] 变成了 [[A, A, A], [B, B, B]]。
💡 核心精髓：PyTorch 的 expand 是一个视图（View）操作，它不会在内存中真正复制数据！
它只是通过修改张量的步长（stride=0）让同一个内存地址在逻辑上被读取了多次，极大地节省了显存。

第三步：reshape(...)
.reshape(batch_size, n_kv_head * n_rep, seq_len, d_k) -> 形状变为: [batch_size, n_head, seq_len, d_k]
作用：将 n_kv_head 和 n_rep 维度合并，因为 n_kv_head * n_rep == n_head。
把 [[A, A, A], [B, B, B]] 展平，变成了 [A, A, A, B, B, B]。

总结：
利用 unsqueeze -> expand -> reshape 的组合技，在不增加额外物理显存的情况下，
巧妙地完成了 K/V 头的逻辑复制与对齐，是 GQA 能够高效运行的关键代码。
"""
