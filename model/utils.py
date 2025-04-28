import torch
from torch import nn


def make_4d_1d(input: torch.Tensor):
    return input.view(-1, input.shape[-4], input.shape[-3], input.shape[-2] * input.shape[-1])


def make_3d_1d(input: torch.Tensor):
    return input.view(-1, input.shape[-3] * input.shape[-2], input.shape[-1])


class BatchNorm1d(nn.Module):
    def __init__(self, alpha: float = 0.5, training: bool = False, input_len: int = 256):
        super(BatchNorm1d, self).__init__()
        self.norm = torch.nn.BatchNorm1d(input_len)
        # self.alpha = alpha
        # self.EMA = torch.tensor(0.)  # agrigator for moving average
        # self.EMS = torch.tensor(1.)  # agrigator for moving scale
        # self.gamma = nn.Parameter(torch.tensor(1.), requires_grad=True)  # learnable scale
        # self.betta = nn.Parameter(torch.tensor(0.), requires_grad=True)  # learnable bias

    def forward(self, batch: torch.Tensor):
        return self.norm(batch)
        # M = torch.mean(batch, dim=(0, 1))

        # if self.training:
        #     self.EMA = self.alpha * self.EMA + (1 - self.alpha) * M

        # batch = batch - self.EMA
        # std = torch.mean(torch.std(batch, dim=1), dim=0)

        # if self.training:
        #     self.EMS = self.alpha * self.EMA + (1 - self.alpha) * std

        # return (batch / self.EMS) * self.gamma + self.betta


if __name__ == '__main__':
    BN = BatchNorm1d(0.3, training=True)
    x = torch.stack([torch.tensor(list(range(10))) for _ in range(5)]).to(torch.float)
    BN(x)

        
