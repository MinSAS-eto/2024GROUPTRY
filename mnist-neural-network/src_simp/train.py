import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import struct
import numpy as np
import matplotlib.pyplot as plt
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

# 定义训练和测试数据集路径
train_images_path = "C:/Users/MinSA/Documents/Academic/Code/2024GROUPTRY/mnist-neural-network/data/mnist-dataset/versions/1/train-images.idx3-ubyte"
train_labels_path = "C:/Users/MinSA/Documents/Academic/Code/2024GROUPTRY/mnist-neural-network/data/mnist-dataset/versions/1/train-labels.idx1-ubyte"
test_images_path = "C:/Users/MinSA/Documents/Academic/Code/2024GROUPTRY/mnist-neural-network/data/mnist-dataset/versions/1/t10k-images.idx3-ubyte"
test_labels_path = "C:/Users/MinSA/Documents/Academic/Code/2024GROUPTRY/mnist-neural-network/data/mnist-dataset/versions/1/t10k-labels.idx1-ubyte"

# 训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')
    return train_loss / len(train_loader)

# 验证函数
def validate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return test_loss, accuracy

# 计算数据集的均值和标准差
temp_transform = transforms.ToTensor()
dataset = MNISTDataset(train_images_path, train_labels_path, transform=temp_transform)
data = torch.stack([img for img, _ in dataset])
data = data.view(data.size(0), -1)
mean = data.mean().item()
std = data.std().item()
print(f"mean: {mean}, std: {std}")

# 主函数
def main():
    # 定义数据预处理和增强
    transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,)),
    transforms.Lambda(lambda x: x.view(-1))
    ])

    # 加载训练数据集
    train_dataset = MNISTDataset(train_images_path, train_labels_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = MNISTDataset(test_images_path, test_labels_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型、优化器和学习率调度器
    model = MLP().to(device)
    epochs = 100
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 记录训练和验证损失
    train_losses = []
    test_losses = []
    test_accuracies = []
    best_test_loss = float('inf')
    patience = 10
    trigger_times = 0

    # 训练模型
    for epoch in range(1, epochs+1):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss, test_accuracy = validate(model, device, test_loader)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        scheduler.step()

        print(f'Epoch {epoch}: Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, Test Accuracy: {test_accuracy:.6f}')

        # 提前停止
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            trigger_times = 0
            torch.save(model.state_dict(), "C:/Users/MinSA/Documents/Academic/Code/2024GROUPTRY/mnist-neural-network/src/model/mlp_mnist.pt")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping!')
                break

    # 绘制训练和验证损失曲线
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curve.png')

    # 绘制验证准确率曲线
    plt.figure()
    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_curve.png')

if __name__ == '__main__':
    main()