
import torch
import torch.nn as nn


class GRPO:
    def __init__(self, clip_epsilon, beta):
        self.clip_epsilon = clip_epsilon
        self.beta = beta
    
    def compute_group_advantages(self, rewards, eps = 1e-8):
        means = rewards.mean(dim = -1, keepdim = True)
        # unbiased=True: 使用无偏估计计算标准差，分母为 N-1 而非 N，
        # 在样本量较小时能更准确地估计总体标准差
        std = rewards.std(dim = -1, keepdim = True, unbiased = True)
        advantages = (rewards - means) / (std + eps)
        # detach(): 将 advantages 从计算图中分离，使其不参与梯度反向传播。
        # 优势值仅作为常数权重使用，不应影响策略网络之外的梯度计算
        return advantages.detach()
    
    def compute_loss(self, rewards, cur_log_probs, old_log_probs, ref_log_probs, mask):
        advantages = self.compute_group_advantages(rewards).unsqueeze(-1) # [B, G, 1]
        importance_ratios = torch.exp(cur_log_probs - old_log_probs) # [B, G, T]
        # clamp 和 clip 在 PyTorch 中完全等价，torch.clip 是 torch.clamp 的别名。
        # 此处将重要性比率限制在 [1-ε, 1+ε] 范围内，防止策略更新步幅过大
        clamped_importances_ratios = torch.clamp(importance_ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        surr1 = importance_ratios * advantages
        surr2 = clamped_importances_ratios * advantages
        ppo_loss = -torch.min(surr1, surr2)

        # log_ratio = log(p_ref / p_cur)，即参考模型与当前模型的对数概率之差
        log_ratio = ref_log_probs - cur_log_probs
        # KL 散度近似公式：KL(cur || ref) ≈ exp(log_ref - log_cur) - (log_ref - log_cur) - 1 ≥ 0
        # 作为惩罚项加到 loss 上（+ beta * kl_penalty），防止当前策略偏离参考模型过远
        kl_penalty = torch.exp(log_ratio) - log_ratio - 1
        per_token_loss = ppo_loss + self.beta * kl_penalty

        # ---- Option 1: Per-token level aggregation (simple, but long sequences have larger weight) ----
        masked_loss = per_token_loss * mask
        mean_loss = masked_loss.sum() / (mask.sum() + 1e-8)

        # ---- Option 2: Original GRPO aggregation: token avg -> group avg -> batch avg ----
        # Step 1: average over steps for each output (normalize sequence length)
        # token_sum = (per_token_loss * mask).sum(dim=-1)   # [B, G]
        # token_count = mask.sum(dim=-1).clamp(min=1e-8)    # [B, G]
        # per_output_loss = token_sum / token_count          # [B, G]

        # Step 2: average over group, then over batch
        # mean_loss = per_output_loss.mean()                 # scalar
        return mean_loss
        