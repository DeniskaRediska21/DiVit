import torch
from torch import nn
from model.utils import BatchNorm1d


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
        attention_matrix = torch.softmax(attention_matrix, dim = 1)

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


class EncoderBlock(nn.Module):
    def __init__(self, n_heads: int = 1, input_len: int = 256, hidden_size: int = 128, training: bool = False, alpha: float = 0.5, n_linear: int = 1):
        super(EncoderBlock, self).__init__()
        self.batch_norm1 = BatchNorm1d(alpha=alpha, training=training)
        self.batch_norm2 = BatchNorm1d(alpha=alpha, training=training)
        self.linears = nn.ModuleList([nn.Linear(input_len, input_len) for _ in range(n_linear)])
        self.activations = nn.ModuleList([nn.ReLU() for _ in range(n_linear)])
        self.attention = MultiheadedAttention(n_heads=n_heads, input_len=input_len, hidden_size=hidden_size)

    def forward(self, input):
        input = self.batch_norm1(input)
        dE = self.attention(input)
        input += dE
        input = self.batch_norm2(input)

        for idx in range(len(input)):
            for linear, activation in zip(self.linears, self.activations):
                input[idx] = activation(linear(input[idx]))
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

