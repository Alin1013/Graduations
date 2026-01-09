# 生成剪枝后的模型
import os
import torch
import torch.nn.utils.prune as prune
import numpy as np
import argparse
from nets.yolov8 import YOLOv8
from utils.utils import get_anchors


def prune_model(model, pruning_ratio=0.3, method='l1_unstructured'):
    """
    对模型卷积层进行剪枝
    
    Args:
        model: 要剪枝的模型
        pruning_ratio: 剪枝比例（如0.3表示剪枝30%的权重）
        method: 剪枝方法 ('l1_unstructured' 或 'ln_structured')
    
    Returns:
        剪枝后的模型
    """
    model.eval()  # 设置为评估模式
    
    pruned_layers = 0
    total_params = 0
    pruned_params = 0
    
    # 遍历所有模块，对卷积层应用剪枝
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # 计算参数数量
            num_params = module.weight.numel()
            total_params += num_params
            
            # 对权重进行剪枝
            if method == 'l1_unstructured':
                # L1非结构化剪枝（会创建稀疏矩阵，但不会真正减少模型大小）
                prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
            elif method == 'ln_structured':
                # Ln结构化剪枝（按通道剪枝，可以真正减少模型大小）
                # 注意：这需要指定dim和n参数
                prune.ln_structured(module, name='weight', amount=pruning_ratio, n=2, dim=0)
            else:
                raise ValueError(f"不支持的剪枝方法: {method}")
            
            # 永久移除被剪枝的权重（将mask应用到参数）
            prune.remove(module, 'weight')
            
            pruned_layers += 1
            pruned_params += int(num_params * pruning_ratio)
    
    print(f"✅ 剪枝完成：")
    print(f"   - 剪枝层数：{pruned_layers}")
    print(f"   - 总参数数：{total_params:,}")
    print(f"   - 剪枝参数数：{pruned_params:,} ({pruning_ratio*100:.1f}%)")
    print(f"   - 剩余参数数：{total_params - pruned_params:,}")
    
    return model


def load_model_config(config_path='model_data/gesture.yaml'):
    """从配置文件加载模型参数"""
    import yaml
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    num_classes = config.get('nc', 18)
    return num_classes


# 剪枝流程示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv8模型剪枝工具')
    parser.add_argument('--model_path', type=str, default='temp/best_model.pt',
                        help='预训练模型权重路径')
    parser.add_argument('--output_path', type=str, default='temp/pruned_model.pt',
                        help='剪枝后模型保存路径')
    parser.add_argument('--anchors_path', type=str, default='yolo_anchors.txt',
                        help='anchors文件路径')
    parser.add_argument('--config_path', type=str, default='model_data/gesture.yaml',
                        help='模型配置文件路径')
    parser.add_argument('--pruning_ratio', type=float, default=0.3,
                        help='剪枝比例（0.0-1.0，如0.3表示剪枝30%%）')
    parser.add_argument('--method', type=str, default='l1_unstructured',
                        choices=['l1_unstructured', 'ln_structured'],
                        help='剪枝方法')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='运行设备')
    
    opt = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(opt.model_path):
        raise FileNotFoundError(f"❌ 模型文件不存在: {opt.model_path}")
    if not os.path.exists(opt.anchors_path):
        raise FileNotFoundError(f"❌ Anchors文件不存在: {opt.anchors_path}")
    
    # 创建输出目录
    output_dir = os.path.dirname(opt.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"✅ 创建输出目录: {output_dir}")
    
    # 加载配置
    print("=" * 60)
    print("YOLOv8 模型剪枝工具")
    print("=" * 60)
    num_classes = load_model_config(opt.config_path)
    print(f"📊 类别数: {num_classes}")
    
    # 加载anchors
    anchors, num_anchors = get_anchors(opt.anchors_path)
    anchors = anchors.tolist()  # 转换为列表格式
    print(f"📊 Anchors数量: {num_anchors}")
    print(f"📊 Anchors: {anchors}")
    
    # 设置设备
    device = torch.device(opt.device if torch.cuda.is_available() and opt.device == 'cuda' else 'cpu')
    print(f"📊 运行设备: {device}")
    print("=" * 60)
    
    # 初始化模型
    print(f"\n🔄 初始化模型...")
    model = YOLOv8(
        num_classes=num_classes,
        anchors=anchors,
        input_shape=[640, 640],
        cuda=(device.type == 'cuda')
    )
    
    # 加载预训练权重
    print(f"🔄 加载预训练模型: {opt.model_path}")
    try:
        state_dict = torch.load(opt.model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("✅ 模型权重加载成功")
    except Exception as e:
        raise RuntimeError(f"❌ 模型权重加载失败: {e}")
    
    # 将模型移到指定设备
    model = model.to(device)
    model.eval()  # 设置为评估模式
    
    # 剪枝
    print(f"\n🔄 开始剪枝（比例: {opt.pruning_ratio*100:.1f}%, 方法: {opt.method}）...")
    pruned_model = prune_model(model, pruning_ratio=opt.pruning_ratio, method=opt.method)
    
    # 保存剪枝后的模型
    print(f"\n🔄 保存剪枝后的模型: {opt.output_path}")
    torch.save(pruned_model.state_dict(), opt.output_path)
    print("✅ 剪枝后的模型已保存")
    
    print("\n" + "=" * 60)
    print("🎉 剪枝流程完成！")
    print("=" * 60)
    print(f"\n⚠️  注意：")
    print(f"   1. 非结构化剪枝（l1_unstructured）会创建稀疏矩阵，")
    print(f"      但不会真正减少模型大小，需要专门的推理引擎支持")
    print(f"   2. 结构化剪枝（ln_structured）可以真正减少模型大小")
    print(f"   3. 剪枝后建议进行微调训练以恢复性能")