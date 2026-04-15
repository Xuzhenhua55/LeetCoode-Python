import torch
import torch.nn.functional as F


class DPO:
    def __init__(self, beta):
        # beta: KL penalty coefficient
        self.beta = beta

    def compute_loss(self, policy_chosen_logps, policy_rejected_logps,
                     ref_chosen_logps, ref_rejected_logps):
        """
        policy_chosen_logps:   [B]  current policy sequence log prob for chosen
        policy_rejected_logps: [B]  current policy sequence log prob for rejected
        ref_chosen_logps:      [B]  reference model sequence log prob for chosen
        ref_rejected_logps:    [B]  reference model sequence log prob for rejected
        """
        # log ratio: log π(y_w|x) - log π_ref(y_w|x)
        chosen_log_ratio   = policy_chosen_logps   - ref_chosen_logps    # [B]
        # log ratio: log π(y_l|x) - log π_ref(y_l|x)
        rejected_log_ratio = policy_rejected_logps - ref_rejected_logps  # [B]

        # DPO loss: -log σ(β * (chosen_log_ratio - rejected_log_ratio))
        logits = self.beta * (chosen_log_ratio - rejected_log_ratio)  # [B]
        loss = -F.logsigmoid(logits)                                  # [B]

        return loss.mean()
