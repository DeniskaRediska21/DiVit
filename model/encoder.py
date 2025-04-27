import torch
from torch import nn
from model.utils import BatchNorm1d


class Attention(nn.Module):
    def __init__(self, input_len: int = 256, hidden_size: int = 128):
        super(Attention, self).__init__()
        self.Q = nn.Parameter(torch.randn((input_len, hidden_size)), requires_grad=True)
        self.K = nn.Parameter(torch.randn((input_len, hidden_size)), requires_grad=True)
        self.V_down = nn.Parameter(torch.randn((input_len, hidden_size)), requires_grad=True)
        self.V_up = nn.Parameter(torch.randn((hidden_size, input_len)), requires_grad=True)
    def forward(self, input):
        query = input @ self.Q
        key = input @ self.K
        V = self.V_down @ self.V_up
        input = input @ V

        attention_matrix = query @ key.transpose(-2, -1)
        attention_matrix = attention_matrix.softmax(dim=-1)

        input = attention_matrix @ input
        return input


class MultiheadedAttention(nn.Module):
    def __init__(self, n_heads: int = 1, input_len: int = 256, hidden_size: int = 128):
        super(MultiheadedAttention, self).__init__()
        self.attention_blocks = nn.ModuleList([Attention(input_len=input_len, hidden_size=hidden_size) for _ in range(n_heads)])

    def forward(self, input):
        dE = self.attention_blocks[0](input)
        for i in range(len(self.attention_blocks) - 1):
            dE = dE + self.attention_blocks[i + 1](input)
        return dE


class EncoderBlock(nn.Module):
    def __init__(self, n_heads: int = 1, input_len: int = 256, hidden_size: int = 128, training: bool = False, alpha: float = 0.5, n_linear: int = 1):
        super(EncoderBlock, self).__init__()
        self.batch_norm1 = BatchNorm1d(alpha=alpha, training=training, input_len=input_len)
        self.batch_norm2 = BatchNorm1d(alpha=alpha, training=training, input_len=input_len)
        self.attention = MultiheadedAttention(n_heads=n_heads, input_len=input_len, hidden_size=hidden_size)

        modules = []
        for _ in range(n_linear):
            modules.append(nn.Linear(input_len, input_len))
            modules.append(nn.ReLU())

        self.MLP = nn.Sequential(*modules)

    def forward(self, input):
        input = self.batch_norm1(input)
        dE = self.attention(input)
        input = input + dE
        input = self.batch_norm2(input)
        input = self.MLP(input)
        return input

class Encoder(nn.Module):
    def __init__(self, n_blocks: int = 1, n_heads: int = 1, input_len: int = 256, hidden_size: int = 128, training: bool = False, alpha: float = 0.5, n_linear: int = 1):
        super(Encoder, self).__init__()
        self.blocks = nn.ModuleList([EncoderBlock(n_heads=n_heads, input_len=input_len, hidden_size=hidden_size, training=training, alpha=alpha, n_linear=n_linear) for _ in range(n_blocks)])

    def forward(self, input):
        for block in self.blocks:
            input = block(input)
        return input


if __name__ == '__main__':
    from utils import make_3d_1d, make_4d_1d
    from positional_embedding import PositionalEmbeddingLearnable

    p_embed = PositionalEmbeddingLearnable()
    x = torch.rand((10, 10, 16, 16))
    x = make_4d_1d(x)
    pos = p_embed(x)
    x = x + pos
    x = make_3d_1d(x)
    encoder = Encoder(n_blocks=1, n_heads=2, training=False, n_linear=3)
    x = encoder(x)

    breakpoint()

