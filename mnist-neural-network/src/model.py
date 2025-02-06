import torch
class CNNNet(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.dropout = torch.nn.Dropout(0.1)
        
        # 使用自适应池化，将特征图固定为 7x7
        self.adaptpool = torch.nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = torch.nn.Linear(128 * 7 * 7, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
        x = self.adaptpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x, dim=1)