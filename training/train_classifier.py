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
    accuracy = 0
    num = 0
    for index, (image, target) in enumerate(pbar):
        image = image.cuda()
        target = target.cuda()

        if training:
            optimizer.zero_grad()
            logits = model(image)
            loss = loss_func(logits, target)
        else:
            with torch.no_grad():
                logits = model(image)
                loss = loss_func(logits, target)

        with torch.no_grad():
            pred = logits.softmax(dim=1).argmax(dim=1)
            accuracy += torch.sum(pred == target).detach().cpu().numpy()
            num += len(target)

        if training:
            loss.backward()
            optimizer.step()
            pbar.set_description(f'Training, loss {loss:.2f}, acc {accuracy / num: .1%}:')
        else:
            pbar.set_description(f'Validation, loss {loss:.2f}, acc {accuracy / num: .1%}:')



LEARNING_RATE = 1e-4
EPOCHS = 20

train_dataloader, val_dataloader = get_MNIST_dataloader(100, shuffle=True)

model = DiVitClassifier(n_blocks=2, n_heads=3, n_class_tokens=1, hidden_size=256, n_channels=1, patch_size=16, n_classes=10, n_linear=1, classifier_layers=4, num_embeddings=20).cuda()

loss = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

for _ in range(EPOCHS):
    epoch(model, train_dataloader, optimizer=optimizer, loss_func=loss, training=True)
    epoch(model, val_dataloader, optimizer=optimizer, loss_func=loss, training=False)




