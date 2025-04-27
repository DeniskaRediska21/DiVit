import torch
import torchvision


def get_MNIST_dataloader(batch_size, shuffle: bool = True):
    transform=torchvision.transforms.v2.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('MNIST', train=True, download=True, transform=transform), shuffle=shuffle, batch_size=batch_size)
    val_dataloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('MNIST', train=False, download=True, transform=transform), shuffle=shuffle, batch_size=batch_size)

    return train_dataloader, val_dataloader
