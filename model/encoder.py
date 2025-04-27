import torch
from torch import nn

class Attention(nn.Module):
    def __init__(self, input_len: int = 256, hidden_size: int = 128):
        super(Attention, self).__init__()
        self.Q = nn.Parameter(torch.randn((input_len, hidden_size)))
        self.K = nn.Parameter(torch.randn((input_len, hidden_size)))
        self.V_down = nn.Parameter(torch.randn((input_len, hidden_size)))
        self.V_up = nn.Parameter(torch.randn((hidden_size, input_len)))
    def forward(self, input):
        query = input.matmul(self.Q)
        key = input.matmul(self.K)
        attention_matrix = query.matmul(key.T)

        dE = (attention_matrix @ (input @ self.V_down)) @ self.V_up
        return dE

class MultiheadedAttention(nn.Module):
    def __init__(self, n_heads: int = 1, input_len: int = 256, hidden_size: int = 128):
        super(MultiheadedAttention, self).__init__()
        self.attention_blocks = nn.ModuleList([Attention(input_len=input_len, hidden_size=hidden_size) for _ in range(n_heads)])

    def forward(self, input):
        dE = self.attention_blocks[0](input)
        for i in range(len(self.attention_blocks) - 1):
            dE += self.attention_blocks[i + 1](input)
        return dE


if __name__ == '__main__':
    from utils import make_3d_1d, make_4d_1d
    from positional_embedding import PositionalEmbeddingLearnable

    p_embed = PositionalEmbeddingLearnable()
    x = torch.rand((10, 10, 16, 16))
    x = make_4d_1d(x)
    x = p_embed(x)
    x = make_3d_1d(x)
    attention = MultiheadedAttention(n_heads=5)
    dE = attention(x)
    x = x + dE

    breakpoint()

