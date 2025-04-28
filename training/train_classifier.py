import torch
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from data.dataloader import get_MNIST_dataloader

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from model.divit import DiVitClassifier

def epoch(model, dataloader, optimizer: None = None, loss_func: None = None, training: bool = True):
    pbar = tqdm(dataloader)
    for image, target in pbar:
        image = image.cuda()
        target = target.cuda()

        if training:
            optimizer.zero_grad()
            logits = model(image)
            loss = loss_func(logits, target)
            print(logits.softmax(dim=1).argmax(dim=1).detach().cpu().numpy(), target.detach().cpu().numpy())
            # print(logits.detach().cpu().numpy())
        else:
            with torch.no_grad():
                logits = model(image)
                loss = loss_func(logits.unsqueeze(0), target)

        if training:
            loss.backward()
            optimizer.step()
            pbar.set_description(f'Training, loss {loss}:')
        else:
            pbar.set_description(f'Validation, loss {loss}:')



torch.autograd.set_detect_anomaly(True)
LEARNING_RATE = 1e-3
EPOCHS = 20

train_dataloader, val_dataloader = get_MNIST_dataloader(100)

model = DiVitClassifier(n_blocks=1, n_heads=5, n_class_tokens=1, hidden_size=128, conv_kernel_size=5, n_channels=1, patch_size=14, n_classes=10, n_linear=1, classifier_layers=3, num_embeddings=10).cuda()

loss = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

for _ in range(EPOCHS):
    epoch(model, train_dataloader, optimizer=optimizer, loss_func=loss, training=True)
    epoch(model, val_dataloader, optimizer=optimizer, loss_func=loss, training=False)




