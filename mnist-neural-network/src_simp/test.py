import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
from sklearn.metrics import confusion_matrix
from model import MLP
from train import MNISTDataset, test_images_path, test_labels_path, mean, std

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

#对测试集进行推理，收集所有预测和真实标签，并计算混淆矩阵
def compute_confusion_matrix(model, device, test_loader):
    
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    cm = confusion_matrix(all_targets, all_preds)
    return cm, all_targets, all_preds

#绘制混淆矩阵的热图
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d",
                cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

#从测试集中随机提取一定数量的分类错误样本，并展示图像、预测标签和真实标签
def show_misclassified_images(model, device, test_loader, num_images=10):
    model.eval()
    misclassified = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)

            for i in range(data.size(0)):
                if preds[i] != target[i]:
                    misclassified.append((data[i].cpu(), preds[i].cpu().item(), target[i].cpu().item()))

    if len(misclassified) > 0:
        misclassified = random.sample(misclassified, min(num_images, len(misclassified)))

    # 绘制错误分类图像
    plt.figure(figsize=(15, 5))
    for i, (img, pred_label, true_label) in enumerate(misclassified):
        if img.dim() == 1:
            img = img.view(28, 28)
        elif img.dim() == 3 and img.size(0) == 1:
            img = img.squeeze(0)
        plt.subplot(1, num_images, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f"P: {pred_label}\nT: {true_label}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 配置测试数据的预处理，与训练时一致
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    # 加载测试数据
    test_dataset = MNISTDataset(test_images_path, test_labels_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = MLP().to(device)
    model.load_state_dict(torch.load("C:/Users/MinSA/Documents/Academic/Code/2024GROUPTRY/mnist-neural-network/src/model/mlp_mnist.pt"))
    
    # 先进行基本测试
    test(model, device, test_loader)
    
    # 计算混淆矩阵并绘制热图
    cm, all_targets, all_preds = compute_confusion_matrix(model, device, test_loader)
    classes = [str(i) for i in range(10)]
    plot_confusion_matrix(cm, classes, normalize=True, title='Normalized Confusion Matrix')
    
    # 展示错误分类的图像示例
    show_misclassified_images(model, device, test_loader, num_images=10)

if __name__ == '__main__':
    main()