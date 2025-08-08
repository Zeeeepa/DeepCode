"""
模型配置文件
"""

# 模型配置
MODEL_CONFIG = {
    "num_classes": 10,
    "backbone": "resnet18",
    "attention_type": "channel",  # 可能需要改为空间注意力或混合注意力
    "dropout_rate": 0.1
}

# 训练配置
TRAIN_CONFIG = {
    "batch_size": 128,  # 可能需要调整
    "learning_rate": 0.001,  # 可能需要优化
    "weight_decay": 1e-4,
    "epochs": 200,
    "optimizer": "adam",  # 可能需要尝试其他优化器
    "scheduler": None  # 缺少学习率调度器
}

# 数据配置
DATA_CONFIG = {
    "dataset": "cifar10",
    "data_dir": "./data",
    "num_workers": 2,
    "pin_memory": True
}
