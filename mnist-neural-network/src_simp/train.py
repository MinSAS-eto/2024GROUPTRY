import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import struct
import numpy as np
from model import MLP
import torchvision.transforms as transforms


# 自定义数据集类
class MNISTDataset(Dataset):
    def __init__(self, image_file, label_file, transform=None):
        self.images = self._read_images(image_file)
        self.labels = self._read_labels(label_file)
        self.transform = transform

    def _read_images(self, filepath):
        with open(filepath, 'rb') as f:
            _, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
        return data

    def _read_labels(self, filepath):
        with open(filepath, 'rb') as f:
            struct.unpack('>II', f.read(8))
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

train_images_path = "C:/Users/MinSA/Documents/Academic/Code/2024GROUPTRY/mnist-neural-network/data/mnist-dataset/versions/1/train-images.idx3-ubyte"
train_labels_path = "C:/Users/MinSA/Documents/Academic/Code/2024GROUPTRY/mnist-neural-network/data/mnist-dataset/versions/1/train-labels.idx1-ubyte"
test_images_path = "C:/Users/MinSA/Documents/Academic/Code/2024GROUPTRY/mnist-neural-network/data/mnist-dataset/versions/1/t10k-images.idx3-ubyte"
test_labels_path = "C:/Users/MinSA/Documents/Academic/Code/2024GROUPTRY/mnist-neural-network/data/mnist-dataset/versions/1/t10k-labels.idx1-ubyte"

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

temp_transform = transforms.ToTensor()
dataset = MNISTDataset(train_images_path, train_labels_path, transform=temp_transform)
data = torch.stack([img for img, _ in dataset])
data = data.view(data.size(0), -1)
mean = data.mean().item()
std = data.std().item()
print(f"mean: {mean}, std: {std}")

def main():
    transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,)),
    transforms.Lambda(lambda x: x.view(-1))
    ])

    train_dataset = MNISTDataset(train_images_path, train_labels_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    model = MLP().to(device)
    epochs = 50
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


    for epoch in range(1, epochs+1):
        train(model, device, train_loader, optimizer, epoch)
        scheduler.step()
    torch.save(model.state_dict(), "C:/Users/MinSA/Documents/Academic/Code/2024GROUPTRY/mnist-neural-network/src/model/mlp_mnist.pt")

if __name__ == '__main__':
    main()