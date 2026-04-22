import torch
import torch.nn as nn


class PPO:
    def __init__(self, clip_epsilon, gamma, lam):
        self.clip_epsilon = clip_epsilon
        # gamma: 折扣因子，控制未来奖励的衰减程度，通常取 0.99
        self.gamma = gamma
        # lam (λ): GAE 平滑系数，控制 bias-variance tradeoff
        # λ=0 退化为 TD(0)（低方差高偏差），λ=1 退化为 MC（高方差低偏差）
        self.lam = lam

    def compute_gae(self, rewards, values):
        """
        Generalized Advantage Estimation (GAE)
        rewards: [B, T]  每个时间步的即时奖励
        values:  [B, T]  当前时间步的状态价值估计 V(s_t)
        """
        # 内部构造 next_values：将 values 向左移一位，最后一步补 0（episode 结束）
        next_values = torch.cat([values[:, 1:], torch.zeros_like(values[:, :1])], dim=-1)  # [B, T]

        # TD 残差：δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        td_errors = rewards + self.gamma * next_values - values  # [B, T]

        B, T = td_errors.shape
        advantages = torch.zeros_like(td_errors)  # [B, T]

        # 从后往前递推计算 GAE：A_t = δ_t + γλ * A_{t+1}
        gae = torch.zeros_like(td_errors[:, 0])
        for t in reversed(range(T)):
            gae = td_errors[:, t] + self.gamma * self.lam * gae
            advantages[:, t] = gae

        # detach(): 优势值作为常数权重，不参与梯度反向传播
        return advantages.detach()

    def compute_policy_loss(self, rewards, values, cur_log_probs, old_log_probs, mask):
        """
        rewards:       [B, T]
        values:        [B, T]   critic 输出的状态价值
        cur_log_probs: [B, T]   当前策略的 log 概率
        old_log_probs: [B, T]   采样时旧策略的 log 概率
        mask:          [B, T]   有效 token 掩码
        """
        advantages = self.compute_gae(rewards, values)  # [B, T]
        # ---- Actor Loss (Clipped Surrogate Objective) ----
        importance_ratios = torch.exp(cur_log_probs - old_log_probs)  # [B, T]
        # clamp 和 clip 在 PyTorch 中完全等价，torch.clip 是 torch.clamp 的别名。
        # 此处将重要性比率限制在 [1-ε, 1+ε] 范围内，防止策略更新步幅过大
        clamped_importance_ratios = torch.clamp(importance_ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        surr1 = importance_ratios * advantages
        surr2 = clamped_importance_ratios * advantages
        per_token_actor_loss = -torch.min(surr1, surr2)  # [B, T]

        # ---- Loss Aggregation ----
        # ---- Option 1: Per-token level aggregation (simple, but long sequences have larger weight) ----
        token_count = mask.sum() + 1e-8
        mean_actor_loss = (per_token_actor_loss * mask).sum() / token_count

        # ---- Option 2: Token avg -> batch avg (normalize by sequence length per sample) ----
        # Step 1: average over tokens for each sample (normalize sequence length)
        # token_sum_actor = (per_token_actor_loss * mask).sum(dim=-1)   # [B]
        # token_count = mask.sum(dim=-1).clamp(min=1e-8)                # [B]
        # per_sample_actor_loss = token_sum_actor / token_count         # [B]

        # Step 2: average over batch
        # mean_actor_loss = per_sample_actor_loss.mean()                # scalar

        return mean_actor_loss

    def compute_value_loss(self, values, returns, mask):
        """
        values:  [B, T]   critic 输出的状态价值
        returns: [B, T]   目标回报 G_t = advantages + old_values
        mask:    [B, T]   有效 token 掩码
        """
        # ---- Critic Loss (Value Function) ----
        # MSE loss，衡量 critic 对状态价值的估计误差
        per_token_critic_loss = (values - returns) ** 2  # [B, T]

        # ---- Loss Aggregation ----
        # ---- Option 1: Per-token level aggregation (simple, but long sequences have larger weight) ----
        token_count = mask.sum() + 1e-8
        mean_critic_loss = (per_token_critic_loss * mask).sum() / token_count

        # ---- Option 2: Token avg -> batch avg (normalize by sequence length per sample) ----
        # Step 1: average over tokens for each sample (normalize sequence length)
        # token_sum_critic = (per_token_critic_loss * mask).sum(dim=-1)  # [B]
        # token_count = mask.sum(dim=-1).clamp(min=1e-8)                 # [B]
        # per_sample_critic_loss = token_sum_critic / token_count        # [B]

        # Step 2: average over batch
        # mean_critic_loss = per_sample_critic_loss.mean()               # scalar

        return mean_critic_loss

    def compute_loss(self, rewards, values, returns,
                     cur_log_probs, old_log_probs, mask):
        """
        rewards:       [B, T]
        values:        [B, T]   critic 输出的状态价值
        returns:       [B, T]   目标回报 G_t = advantages + old_values
        cur_log_probs: [B, T]   当前策略的 log 概率
        old_log_probs: [B, T]   采样时旧策略的 log 概率
        mask:          [B, T]   有效 token 掩码
        """
        mean_actor_loss = self.compute_policy_loss(rewards, values, cur_log_probs, old_log_probs, mask)
        mean_critic_loss = self.compute_value_loss(values, returns, mask)

        # 总 loss = actor loss + critic loss（系数 0.5 为常见超参，防止 critic loss 量级过大）
        total_loss = mean_actor_loss + 0.5 * mean_critic_loss
        return total_loss, mean_actor_loss, mean_critic_loss
