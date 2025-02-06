
class CNNNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNNet, self).__init__()
        # 第一层卷积：输入通道数1，输出32，卷积核3×3，保持尺寸
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 第二层卷积：输入32，输出64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 池化层，2×2降采样
        self.pool = nn.MaxPool2d(2, 2)
        # Dropout用于防止过拟合
        self.dropout = nn.Dropout(0.25)
        # 全连接层，输入特征 64*14*14，输出128维
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        # 输出层
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # 卷积层1 + 激活函数
        x = F.relu(self.conv1(x))
        # 卷积层2 + 激活函数 + 池化
        x = self.pool(F.relu(self.conv2(x)))
        # Dropout
        x = self.dropout(x)
        # 展平操作
        x = x.view(x.size(0), -1)
        # 全连接层1 + 激活函数
        x = F.relu(self.fc1(x))
        # Dropout
        x = self.dropout(x)
        # 输出层
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)