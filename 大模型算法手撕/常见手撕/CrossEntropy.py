import torch


class CrossEntropyLoss:
    def __init__(self, reduction: str = 'mean'):
        # reduction: 'mean' | 'sum' | 'none'，控制 loss 的聚合方式
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  [B, C]  模型输出的原始 logits（未经 softmax）
        targets: [B]     每个样本的真实类别索引（整数标签）
        return:  scalar / [B]  取决于 reduction 参数

        公式推导：
            CE = -log( exp(x_y) / sum(exp(x_j)) )
               = -x_y + log( sum(exp(x_j)) )
               = -x_y + log_sum_exp(x)
        """
        # Step 1: 数值稳定的 softmax
        # 直接计算 exp(logits) 会因数值过大溢出，需先减去每行最大值（不改变 softmax 结果）
        x_max = logits.max(dim=-1, keepdim=True).values           # [B, 1]
        shifted = logits - x_max                                   # [B, C]  平移后最大值为 0
        exp_shifted = torch.exp(shifted)                           # [B, C]
        probs = exp_shifted / exp_shifted.sum(dim=-1, keepdim=True)  # [B, C]  softmax 概率

        # Step 2: 对概率取 log，得到 log-softmax
        # 加 eps 防止 log(0)（理论上 softmax 输出 > 0，但数值上可能极小）
        log_probs = torch.log(probs + 1e-10)                       # [B, C]

        # Step 3: 取出每个样本目标类别对应的 log 概率，取负
        # gather 按 targets 索引从 log_probs 中取出对应位置的值
        # targets.unsqueeze(-1): [B] -> [B, 1]，gather 后 squeeze 回 [B]
        loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # [B]

        # Step 4: 按 reduction 方式聚合
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            # 'none': 返回每个样本的 loss，不做聚合
            return loss


# ── 验证 ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    torch.manual_seed(0)
    B, C = 4, 8
    logits  = torch.randn(B, C)
    targets = torch.randint(0, C, (B,))

    ce_custom = CrossEntropyLoss(reduction='mean')
    ce_torch  = torch.nn.CrossEntropyLoss(reduction='mean')

    loss_custom = ce_custom.forward(logits, targets)
    loss_torch  = ce_torch(logits, targets)

    print(f"custom loss : {loss_custom.item():.6f}")
    print(f"torch  loss : {loss_torch.item():.6f}")
    print(f"diff        : {abs(loss_custom.item() - loss_torch.item()):.2e}")

"""
补充说明：log_probs.gather 的语法含义

在 PyTorch 中，torch.gather 的核心作用是：根据指定的索引（index），沿着指定的维度（dim）从原张量中“收集”或“提取”元素。

代码：loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
目的：从模型预测的每个样本的概率分布中，精准地把“真实标签（target）对应位置的那个概率值”给抽出来。

拆解步骤：
1. targets.unsqueeze(-1)：对齐维度
   gather 要求 index 张量的维度数量必须和原张量完全一致。
   log_probs 是 [B, C] (2维)，targets 是 [B] (1维)。
   unsqueeze(-1) 把 targets 变成 [B, 1]。
   
2. log_probs.gather(dim=-1, index=...)：核心提取逻辑
   dim=-1（即 dim=1）表示沿着列（类别维度）去寻找。
   内部逻辑：对于第 i 行，去取第 index[i][0] 列的值。
   执行 gather 之后，返回的结果形状和 index 一样，是 [B, 1]。
   
3. .squeeze(-1)：降维还原
   gather 返回 [B, 1]，加上 .squeeze(-1) 把多余的维度 1 去掉，变成 [B]。
"""
