import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitMarginL1(nn.Module):
    """Add marginal penalty to logits:
        CE + alpha * max(0, max(l^n) - l^n - margin)
    """
    def __init__(self,
                 margin=10,
                 alpha=0.1,
                 ignore_index=-100,
                 mu=0,
                 max_alpha=100.0,
                 step_size=100):
        super().__init__()
        self.margin = margin
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.mu = mu
        self.max_alpha = max_alpha
        self.step_size = step_size


    def get_diff(self, inputs):
        max_values = inputs.max(dim=1)
        max_values = max_values.values.unsqueeze(dim=1).repeat(1, inputs.shape[1])
        diff = max_values - inputs
        return diff

    def forward(self, inputs, targets):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)

        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]

        diff = self.get_diff(inputs)
        loss_margin = F.relu(diff-self.margin).mean()

        return loss_margin
