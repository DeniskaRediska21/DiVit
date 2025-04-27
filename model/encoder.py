import torch
from torch import nn

class Attention(nn.Module):
    def __init__(self, input_len: int = 256, hidden_size: int = 128):
        super(Attention, self).__init__()
        self.Q = nn.Parameter(torch.randn((input_len, hidden_size)))
        self.K = nn.Parameter(torch.randn((input_len, hidden_size)))
        self.V_up = nn.Parameter(torch.randn((input_len, hidden_size)))
        self.V_down = nn.Parameter(torch.randn((input_len, hidden_size)))
    def forward(self, input):
        query = input.matmul(self.Q)
        key = input.matmul(self.K)
        attention_matrix = query.matmul(key.T)
        breakpoint()
        pass

if __name__ == '__main__':
    from utils import make_3d_1d, make_4d_1d
    from positional_embedding import PositionalEmbeddingLearnable

    p_embed = PositionalEmbeddingLearnable()
    x = torch.rand((10, 10, 16, 16))
    x = make_4d_1d(x)
    x = p_embed(x)
    x = make_3d_1d(x)
    attention = Attention()
    x = attention(x)

