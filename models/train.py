from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import MNISTModel


def main():
    torch.manual_seed(42)
    train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=256)
    model = MNISTModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(3):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/3 complete')

    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            correct += (model(images).argmax(dim=1) == labels).sum().item()
    print(f'Test accuracy: {correct / len(test_data):.2%}')
    torch.save(model.state_dict(), Path(__file__).with_name('mnist_model.pth'))
    print('Saved models/mnist_model.pth')


if __name__ == '__main__':
    main()
