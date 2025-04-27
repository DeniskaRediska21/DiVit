import torch
from torch import nn

from utils import make_3d_1d, make_4d_1d
from positional_embedding import PositionalEmbeddingLearnable
from encoder import Encoder

class DiVit(nn.Module):
    def __init__(self, n_blocks: int = 1, n_heads: int = 1, patch_size: int = 16, hidden_size: int = 128, training: bool = False, alpha: float = 0.5, n_linear: int = 1, num_embeddings: int = 100, n_class_tokens: int = 0, n_channels: int = 3, conv_kernel_size: int = 16):
        super(DiVit, self).__init__()

        self.patch_size = 16
        input_len = self.patch_size ** 2

        self.input_conv = nn.Conv2d(n_channels, 1, (conv_kernel_size, conv_kernel_size))

        self.positional_embed = PositionalEmbeddingLearnable(input_len, num_embeddings=num_embeddings)

        self.encoder = Encoder(n_blocks=n_blocks, n_heads=n_heads, input_len=input_len, hidden_size=hidden_size, training=training, alpha=alpha, n_linear=n_linear)

        self.class_token = None if n_class_tokens == 0 else nn.Parameter(torch.rand((n_class_tokens, input_len)))

    def split_into_patches(self, input):
        return input.unfold(0, self.patch_size, self.patch_size).unfold(1, self.patch_size, self.patch_size).contiguous()

    def forward(self, input):
        input = self.input_conv(input).squeeze(0)
        input = self.split_into_patches(input)
        input = make_4d_1d(input)
        pos = self.positional_embed(input)
        input += pos
        input = make_3d_1d(input)

        input = input if self.class_token is None else torch.vstack([input, self.class_token])  # appends class_token, can get with input[-1]

        input = self.encoder(input)
        return input

class DiVitClassifier(nn.Module):
    def __init__(self, n_blocks: int = 1, n_heads: int = 1, patch_size: int = 16, hidden_size: int = 128, training: bool = False, alpha: float = 0.5, n_linear: int = 1, num_embeddings: int = 100, n_class_tokens: int = 0, n_channels: int = 3, conv_kernel_size: int = 16, classifier_layers: int = 3, n_classes: int = 10):
        super(DiVitClassifier, self).__init__()

        input_len = patch_size ** 2
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
        return self.classifier_head(embeddings[-1])
        

if __name__ == '__main__':
    x = torch.rand((3, 1039, 1039))
    divit_classifier = DiVitClassifier(n_blocks=2, n_heads=10, n_class_tokens=1, hidden_size=2048).cuda()
    logits = divit_classifier(x.cuda())
    breakpoint()

