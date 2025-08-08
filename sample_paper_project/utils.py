"""
工具函数
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

def calculate_accuracy(outputs, labels):
    """计算准确率"""
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return 100 * correct / total

def save_model(model, path):
    """保存模型"""
    torch.save(model.state_dict(), path)
    print(f"模型已保存到: {path}")

def plot_training_curve(losses, accuracies):
    """绘制训练曲线 - 功能不完整"""
    # 这个函数需要完善
    pass

def data_augmentation_policy():
    """数据增强策略 - 可能需要改进"""
    # 当前的数据增强可能不够强
    return None
