import torch
import torch.nn.functional as F


class TopKSampling:
    def __init__(self, k: int):
        # k: 每次只保留概率最高的 k 个 token，其余置为 -inf
        self.k = k

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, V]  模型输出的原始 logits（未经 softmax）
        return: [B]     每个样本采样得到的 token id
        """
        B, V = logits.shape

        # Step 1: 找出每行第 k 大的值作为阈值
        # topk 返回 (values, indices)，values 已按降序排列
        topk_values, _ = torch.topk(logits, self.k, dim=-1)  # [B, k]

        # Step 2: 取第 k 大的值作为截断阈值（最小保留值）
        threshold = topk_values[:, -1].unsqueeze(-1)  # [B, 1]

        # Step 3: 将低于阈值的 logits 置为 -inf，使其 softmax 后概率为 0
        filtered_logits = logits.masked_fill(logits < threshold, float('-inf'))  # [B, V]

        # Step 4: 将 logits 转为概率分布，并按概率采样
        probs = F.softmax(filtered_logits, dim=-1)  # [B, V]

        # torch.multinomial: 按概率分布进行多项式采样，num_samples=1 表示每行采一个
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]
        return next_token


class TopPSampling:
    def __init__(self, p: float):
        # p: 累积概率阈值（nucleus），通常取 0.9 / 0.95
        # 只保留累积概率恰好超过 p 的最小 token 集合（nucleus）
        self.p = p

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, V]  模型输出的原始 logits（未经 softmax）
        return: [B]     每个样本采样得到的 token id
        """
        # Step 1: 先将 logits 转为概率分布
        probs = F.softmax(logits, dim=-1)  # [B, V]

        # Step 2: 按概率从大到小排序
        # descending=True 保证累积概率从最高概率 token 开始累加
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)  # [B, V]

        # Step 3: 计算累积概率
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)  # [B, V]

        # Step 4: 构造掩码，将累积概率超过 p 的位置（排除第一个超过的 token）置为 True
        # 向右移一位（shift right）：确保累积概率刚好超过 p 的那个 token 本身被保留
        # 例如 p=0.9，累积到某 token 后首次 >= 0.9，该 token 仍保留，之后的才被过滤
        remove_mask = cumulative_probs - sorted_probs > self.p  # [B, V]

        # Step 5: 将需要过滤的位置置为 -inf（在排序后的空间中操作）
        sorted_logits = torch.full_like(sorted_probs, float('-inf'))
        sorted_logits[~remove_mask] = sorted_probs[~remove_mask].log()  # 转回 log 空间

        # Step 6: 将过滤后的 logits 还原到原始 token 顺序
        # scatter_ 按 sorted_indices 将排序空间的值写回原始位置
        filtered_logits = torch.full_like(logits, float('-inf'))  # [B, V]
        filtered_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)  # [B, V]

        # Step 7: 重新 softmax 归一化后采样
        filtered_probs = F.softmax(filtered_logits, dim=-1)  # [B, V]
        next_token = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)  # [B]
        return next_token


class TopPTopKSampling:
    def __init__(self, k: int, p: float, temperature: float = 1.0):
        # 先做 top-k 截断，再做 top-p nucleus 过滤，最后按温度缩放采样
        # temperature < 1: 分布更尖锐（更确定性）；temperature > 1: 分布更平坦（更随机）
        self.k = k
        self.p = p
        self.temperature = temperature

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, V]  模型输出的原始 logits（未经 softmax）
        return: [B]     每个样本采样得到的 token id
        """
        # Step 1: Temperature scaling — 在 softmax 之前缩放 logits
        # 等价于对概率分布做幂次变换：p_i^(1/T) / sum(p_j^(1/T))
        logits = logits / self.temperature  # [B, V]

        # Step 2: Top-K 过滤 — 只保留概率最高的 k 个 token
        topk_values, _ = torch.topk(logits, self.k, dim=-1)  # [B, k]
        threshold = topk_values[:, -1].unsqueeze(-1)          # [B, 1]
        logits = logits.masked_fill(logits < threshold, float('-inf'))  # [B, V]

        # Step 3: Top-P 过滤 — 在 top-k 结果上进一步做 nucleus 截断
        probs = F.softmax(logits, dim=-1)  # [B, V]
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)  # [B, V]
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)  # [B, V]
        remove_mask = cumulative_probs - sorted_probs > self.p  # [B, V]

        # 将需要移除的位置在排序空间中置为 0，再 scatter 回原始顺序
        sorted_probs[remove_mask] = 0.0
        filtered_probs = torch.zeros_like(probs)
        filtered_probs.scatter_(dim=-1, index=sorted_indices, src=sorted_probs)  # [B, V]

        # Step 4: 重新归一化（top-k/top-p 截断后概率之和不再为 1）
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Step 5: 按最终概率分布采样
        next_token = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)  # [B]
        return next_token
