import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

for device_index in range(torch.cuda.device_count()):
    device = 'cuda:{}'.format(device_index)
    device_name = torch.cuda.get_device_name(device)
    print('{} ({})'.format(device, device_name))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 2048)
        self.fc2 = nn.Linear(2048, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(model, device, train_loader, test_loader, lr=1.0, gamma=0.7):
    print()
    print('Train {}'.format(device))
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, epochs + 1):
        train_one_epoch(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

def train_one_epoch(model, device, train_loader, optimizer, epoch):
    losses = []
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return losses


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset1 = datasets.MNIST('/workspace/data', train=True, download=True, transform=transform)
dataset2 = datasets.MNIST('/workspace/data', train=False, transform=transform)

import timeit

# Input batch size for training
batch_size = 128
# Input batch size for testing
test_batch_size = 1000
# Number of epochs to train
epochs = 1

train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=test_batch_size)

# CPU benchmark
device = 'cpu'
model = Net().to(device)
variables = {
    'model': model, 'device': device, 'train_loader': train_loader, 'test_loader': test_loader,
}
cpu_time = timeit.timeit(f'train(model, device, train_loader, test_loader)', globals=variables, number=1, setup="from __main__ import train")

# GPU benchmark
device = 'cuda:0'
model = Net().to(device)
variables = {
    'model': model, 'device': device, 'train_loader': train_loader, 'test_loader': test_loader,
}
gpu_time = timeit.timeit(f'train(model, device, train_loader, test_loader)', globals=variables, number=1, setup="from __main__ import train")

# Results
print('CPU took {:.2f}s'.format(cpu_time))
print('GPU took {:.2f}s'.format(gpu_time))
print('GPU is {:.1f}x times faster than CPU to train this model'.format(cpu_time / gpu_time))