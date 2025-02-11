import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义自定义激活函数 Mish
class Mish(nn.Module):
    def forward(self, x):
        # Mish 激活函数：x * tanh(softplus(x))
        return x * torch.tanh(F.softplus(x))

# 定义多层感知机（MLP）模型
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(MLP, self).__init__()
        # 定义第一层全连接层
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 使用自定义的 Mish 激活函数
        self.mish = Mish()
        # 定义 Dropout 层，防止过拟合
        self.dropout = nn.Dropout(0.1)
        # 定义第二层全连接层
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # 前向传播：第一层全连接层 -> Mish 激活函数 -> Dropout 层 -> 第二层全连接层
        out = self.fc1(x)
        out = self.mish(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# 测试模型定义
if __name__ == "__main__":
    model = MLP()