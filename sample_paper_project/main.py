#!/usr/bin/env python3
"""
深度学习模型训练脚本
基于论文: "Attention-Enhanced ResNet for Image Classification"
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

class AttentionModule(nn.Module):
    """注意力模块 - 可能需要优化"""
    def __init__(self, channels):
        super().__init__()
        # 简单的通道注意力，可能不够高效
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 16),
            nn.ReLU(),
            nn.Linear(channels // 16, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ResNetWithAttention(nn.Module):
    """带注意力的ResNet模型"""
    def __init__(self, num_classes=10):
        super().__init__()
        # 使用预训练的ResNet作为骨干网络
        self.backbone = torchvision.models.resnet18(pretrained=False)
        
        # 添加注意力模块
        self.attention = AttentionModule(512)
        
        # 修改最后的分类层
        self.backbone.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # ResNet前向传播，但在最后加入注意力
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # 应用注意力机制
        x = self.attention(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)
        
        return x

def train_model():
    """训练模型 - 可能需要优化超参数和训练策略"""
    # 数据预处理 - 可能需要改进数据增强策略
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 加载CIFAR-10数据集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetWithAttention(num_classes=10).to(device)
    
    # 优化器配置 - 可能需要调整学习率和权重衰减
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 训练循环 - 可能需要添加学习率调度器
    print("开始训练...")
    model.train()
    
    for epoch in range(2):  # 只训练2个epoch用于演示
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            if i >= 10:  # 只训练前10个batch用于演示
                break
                
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 5 == 4:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 5:.3f}')
                running_loss = 0.0
    
    print("训练完成！")
    return model

if __name__ == "__main__":
    try:
        model = train_model()
        print("✅ 模型训练成功")
    except Exception as e:
        print(f"❌ 训练失败: {e}")
