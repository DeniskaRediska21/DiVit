import torch
from tqdm import tqdm

from data.dataloader import get_MNIST_dataloader

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from model.divit import DiVitClassifier

train_dataloader, val_dataloader = get_MNIST_dataloader(1)

model = DiVitClassifier(n_blocks=2, n_heads=10, n_class_tokens=1, hidden_size=2048, conv_kernel_size=1, n_channels=1).cuda()

image, target = next(iter(train_dataloader))
ligits = model(image.squeeze(0).cuda())

