import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from model import MLP
from train import MNISTDataset, test_images_path, test_labels_path
from train import mean, std


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # 计算测试损失，并累计
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            # 获取预测类别
            pred = output.argmax(dim=1, keepdim=True)
            # 正确预测的数量
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 配置测试数据的预处理（与训练时一致）
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (std,)), transforms.Lambda(lambda x: x.view(-1))])
    # 加载测试数据
    test_dataset = MNISTDataset(test_images_path, test_labels_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = MLP().to(device)
    model.load_state_dict(torch.load("C:/Users/MinSA/Documents/Academic/Code/2024GROUPTRY/mnist-neural-network/src/model/mlp_mnist.pt"))
    
    test(model, device, test_loader)

if __name__ == '__main__':
    main()