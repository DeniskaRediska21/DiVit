import torch
from torch import nn

class PositionalEmbeddingLearnable(nn.Module):
    def __init__(self, embedding_dim: int = 256, num_embeddings: int = 100):
        super(PositionalEmbeddingLearnable, self).__init__()

        self.embed_row = torch.nn.Embedding(num_embeddings, embedding_dim)
        self.embed_col = torch.nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, input):
        _, h, w, *_ = input.shape

        p_rows = self.embed_row(torch.arange(h, device=input.device))
        p_cols = self.embed_col(torch.arange(w, device=input.device))

        pos = p_cols.unsqueeze(0).repeat(h, 1, 1) + p_rows.unsqueeze(1).repeat(1, w, 1)
        return pos


if __name__ == '__main__':
    from utils import make_4d_1d
    p_embed = PositionalEmbeddingLearnable()
    x = torch.rand((10, 10, 16, 16))
    x = make_4d_1d(x)
    x = p_embed(x)
