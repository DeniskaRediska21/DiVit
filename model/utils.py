import torch
from torch import nn


def make_4d_1d(input: torch.Tensor):
    return input.view(input.shape[0], input.shape[1], input.shape[2] * input.shape[3])


def make_3d_1d(input: torch.Tensor):
    return input.view(input.shape[0] * input.shape[1], input.shape[2])


class BatchNorm1d(nn.Module):
    def __init__(self, alpha: float = 0.5, training: bool = False):
        super(BatchNorm1d, self).__init__()
        self.alpha = alpha
        self.EMA = 0  # agrigator for moving average
        self.EMS = 1  # agrigator for moving scale
        self.gamma = nn.Parameter(torch.tensor(1.), requires_grad=True)  # learnable scale
        self.betta = nn.Parameter(torch.tensor(0.), requires_grad=True)  # learnable bias

    def forward(self, batch: torch.Tensor):
        M = torch.mean(batch, dim=(0, 1))

        if self.training:
            self.EMA = self.alpha * self.EMA + (1 - self.alpha) * M

        batch -= self.EMA
        std = torch.mean(torch.std(batch, dim=1), dim=0)

        if self.training:
            self.EMS = self.alpha * self.EMA + (1 - self.alpha) * std

        return (batch / self.EMS) * self.gamma + self.betta


if __name__ == '__main__':
    BN = BatchNorm1d(0.3, training=True)
    x = torch.stack([torch.tensor(list(range(10))) for _ in range(5)]).to(torch.float)
    BN(x)

        
