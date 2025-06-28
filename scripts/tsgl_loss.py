import torch
import torch.nn as nn

class TSGLoss(nn.Module):
    def __init__(self, beta1=1e-4, beta2=1e-4, beta3=1e-4):
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3

    def forward(self, model, ce_loss):
        W = model.conv2.weight  # (F2, F1*D, 1, 16)
        W = W.squeeze(2)        # (F2, F1*D, 16)
        group_lasso = torch.sum(torch.norm(W, dim=(1,2)))
        sparse_lasso = torch.sum(torch.abs(W))
        temporal_diff = torch.diff(W, dim=2)
        temporal_smooth = torch.sum(torch.norm(temporal_diff, dim=(1,2)))
        return ce_loss + self.beta1 * group_lasso + self.beta2 * sparse_lasso + self.beta3 * temporal_smooth