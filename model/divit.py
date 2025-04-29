import torch
from torch import nn

from model.utils import make_3d_1d
from model.positional_embedding import PositionalEmbeddingLearnable
from model.encoder import Encoder
from model.input_layers import BasicPatches, BasicConv

class DiVit(nn.Module):
    def __init__(self, n_blocks: int = 1, n_heads: int = 1, patch_size: int = 16, hidden_size: int = 128, training: bool = False, alpha: float = 0.5, n_linear: int = 1, num_embeddings: int = 100, n_class_tokens: int = 0, n_channels: int = 3, conv_kernel_size: int = 16):
        super(DiVit, self).__init__()

        self.patch_size = patch_size
        input_len = self.patch_size ** 2

        # self.input_layer = BasicPatches(patch_size)
        self.input_layer = BasicConv(patch_size, n_channels=n_channels)

        self.positional_embed = PositionalEmbeddingLearnable(input_len, num_embeddings=num_embeddings)

        self.encoder = Encoder(n_blocks=n_blocks, n_heads=n_heads, input_len=input_len, hidden_size=hidden_size, training=training, alpha=alpha, n_linear=n_linear)
        # self.encoder = Attention(input_len=input_len, hidden_size=hidden_size)

        self.class_token = None if n_class_tokens == 0 else nn.Parameter(torch.rand((n_class_tokens, input_len)), requires_grad=True)

    def forward(self, input):
        input = self.input_layer(input)
        pos = self.positional_embed(input)
        input = input + pos
        input = make_3d_1d(input)

        input = input if self.class_token is None else torch.hstack([input, self.class_token.repeat(input.shape[0], 1, 1)])  # appends class_token, can get with input[-1]

        input = self.encoder(input)
        return input

class DiVitClassifier(nn.Module):
    def __init__(self, n_blocks: int = 1, n_heads: int = 1, patch_size: int = 16, hidden_size: int = 128, training: bool = False, alpha: float = 0.5, n_linear: int = 1, num_embeddings: int = 100, n_class_tokens: int = 0, n_channels: int = 3, conv_kernel_size: int = 16, classifier_layers: int = 3, n_classes: int = 10):
        super(DiVitClassifier, self).__init__()

        self.patch_size = patch_size
        self.conv_kernel_size = conv_kernel_size
        input_len = patch_size ** 2

        self.layer_norm = nn.LayerNorm(input_len)

        modules = []
        linear_sizes = torch.linspace(input_len, n_classes, steps=classifier_layers + 1, dtype=int)
        for idx in range(len(linear_sizes) - 1):
            modules.append(nn.Linear(linear_sizes[idx], linear_sizes[idx + 1]))
            if idx != len(linear_sizes) - 2:
                modules.append(nn.ReLU())

        self.classifier_head = nn.Sequential(*modules)
        self.vit = DiVit(n_blocks=n_blocks, n_heads=n_heads, patch_size=patch_size, hidden_size=hidden_size, training=training, alpha=alpha, n_linear=n_linear, num_embeddings=num_embeddings, n_class_tokens=n_class_tokens, n_channels=n_channels, conv_kernel_size=conv_kernel_size)

    def forward(self, input):

        embeddings = self.vit(input)
        logits = self.classifier_head(embeddings[:, -1])
        return logits.softmax(dim=1)
        

if __name__ == '__main__':
    x = torch.rand((3, 1039, 1039))
    divit_classifier = DiVitClassifier(n_blocks=2, n_heads=10, n_class_tokens=1, hidden_size=2048).cuda()
    logits = divit_classifier(x.cuda())
    breakpoint()

