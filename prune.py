#生成剪枝后的模型
import torch
import torch.nn.utils.prune as prune
from nets.yolov8 import YOLOv8

def prune_model(model, pruning_ratio=0.3):
    """
    对模型卷积层进行L1剪枝
    pruning_ratio: 裁剪比例（如0.3表示裁剪30%的通道）
    """
    # 遍历所有模块，对卷积层应用L1剪枝
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # 对权重进行L1非结构化剪枝
            prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
            # 永久移除被剪枝的权重（将mask应用到参数）
            prune.remove(module, 'weight')
    return model

# 剪枝流程示例
if __name__ == "__main__":
    # 加载预训练模型
    model = YOLOv8(num_classes=8)
    model.load_state_dict(torch.load("temp/best_model.pt"))  # 加载训练好的模型

    # 剪枝
    pruned_model = prune_model(model, pruning_ratio=0.3)

    # 保存剪枝后的模型
    torch.save(pruned_model.state_dict(), "temp/pruned_model.pt")