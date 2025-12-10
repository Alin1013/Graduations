import numpy as np
from PIL import Image
from get_yaml import get_config
import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

# 忽略无关警告
warnings.filterwarnings('ignore')

from yolo import YOLO


# -------------------------- 新增：YOLO TXT标注生成函数 --------------------------
def generate_yolo_txt(boxes, img_size, save_path, img_name):
    """
    生成YOLO格式的TXT标注文件（替代原XML生成）
    :param boxes: 检测框列表，格式为 [(label, ymin, xmin, ymax, xmax), ...]
    :param img_size: 图像尺寸 (width, height)
    :param save_path: 标注保存根目录
    :param img_name: 图像文件名（不含后缀）
    """
    os.makedirs(save_path, exist_ok=True)
    txt_path = os.path.join(save_path, f"{img_name}.txt")

    # 获取类别映射（从配置文件加载）
    config = get_config()
    class_names = config['names']
    class2id = {name: idx for idx, name in enumerate(class_names)}

    with open(txt_path, 'w', encoding='utf-8') as f:
        for box in boxes:
            try:
                label, ymin, xmin, ymax, xmax = box

                # 过滤无效类别
                if label not in class2id:
                    print(f"⚠️  未知类别 {label}，跳过该框")
                    continue

                # 转换为YOLO格式（归一化中心坐标+宽高）
                img_w, img_h = img_size
                x_center = (xmin + xmax) / 2 / img_w
                y_center = (ymin + ymax) / 2 / img_h
                width = (xmax - xmin) / img_w
                height = (ymax - ymin) / img_h

                # 校验坐标有效性
                if x_center < 0 or x_center > 1 or y_center < 0 or y_center > 1:
                    print(f"⚠️  坐标越界，跳过该框：{box}")
                    continue
                if width <= 0 or height <= 0 or width > 1 or height > 1:
                    print(f"⚠️  宽高无效，跳过该框：{box}")
                    continue

                # 写入TXT（保留6位小数）
                cls_id = class2id[label]
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            except Exception as e:
                print(f"⚠️  处理框 {box} 失败：{e}，跳过")
                continue

    if os.path.getsize(txt_path) == 0:
        # 删除空标注文件
        os.remove(txt_path)
        print(f"⚠️  {img_name}.txt 无有效标注，已删除")
    else:
        print(f"✅ 生成标注：{txt_path}")


# -------------------------- 优化：注意力可视化函数 --------------------------
def visualize_attention(yolo_model, image, save_dir="attention_maps"):
    """
    优化版注意力可视化（适配YOLOv8，兼容更多模型结构）
    :param yolo_model: YOLO模型实例
    :param image: PIL图像对象
    :param save_dir: 注意力图保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    attention_maps = []
    hooks = []

    # 定义通用钩子函数
    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            attention_maps.append(output)
        elif isinstance(output, (tuple, list)):
            # 处理多输出模块
            for out in output:
                if isinstance(out, torch.Tensor):
                    attention_maps.append(out)

    # -------------------------- 适配YOLOv8的注意力模块注册 --------------------------
    try:
        # 方案1：适配YOLOv8 Neck（PAN-FPN）中的注意力模块
        if hasattr(yolo_model.model, 'model'):
            # YOLOv8模型结构：model.model是核心网络
            for idx, module in enumerate(yolo_model.model.model):
                # 查找包含CBAM/注意力的模块（支持多种命名）
                module_name = str(module).lower()
                if any(key in module_name for key in ['cbam', 'attention', 'eca']):
                    hook = module.register_forward_hook(hook_fn)
                    hooks.append(hook)
                    print(f"📌 为模块 {idx} ({module.__class__.__name__}) 注册注意力钩子")

        # 方案2：兼容旧版PANet结构
        if not hooks and hasattr(yolo_model.model, 'neck'):
            neck = yolo_model.model.neck
            for name, module in neck.named_modules():
                if 'cbam' in name.lower() or 'attention' in name.lower():
                    hook = module.register_forward_hook(hook_fn)
                    hooks.append(hook)
                    print(f"📌 为Neck模块 {name} 注册注意力钩子")

        if not hooks:
            print("⚠️  未检测到注意力模块，跳过注意力可视化")
            return

    except AttributeError as e:
        print(f"⚠️  注册注意力钩子失败: {e}")
        return

    # -------------------------- 图像预处理（适配YOLOv8） --------------------------
    try:
        # 转换为模型输入格式
        img_np = np.array(image.convert('RGB'))  # 确保RGB格式
        # 归一化 + 转Tensor + 添加批次维度
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(yolo_model.device)

        # 前向传播触发钩子
        with torch.no_grad():
            yolo_model.model(img_tensor)

        # -------------------------- 保存注意力图 --------------------------
        for idx, am in enumerate(attention_maps):
            try:
                # 处理不同维度的注意力图
                am = am.detach().cpu()
                # 降维：通道平均
                if len(am.shape) == 4:  # [B, C, H, W]
                    am = am.squeeze(0)  # 移除批次维度
                if len(am.shape) == 3:  # [C, H, W]
                    am = am.mean(dim=0)  # 通道平均

                # 归一化到0-1
                am = (am - am.min()) / (am.max() - am.min() + 1e-8)
                # 缩放至原图尺寸
                am_np = am.numpy()
                am_resized = np.array(Image.fromarray(am_np).resize(image.size, Image.BILINEAR))

                # 绘制并保存
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                ax.imshow(image)
                ax.imshow(am_resized, cmap='jet', alpha=0.5)
                ax.axis('off')
                plt.tight_layout(pad=0)
                save_path = os.path.join(save_dir, f"attention_{idx}.png")
                plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
                plt.close(fig)

            except Exception as e:
                print(f"⚠️  处理注意力图 {idx} 失败：{e}")
                continue

        print(f"✅ 注意力图已保存至 {save_dir}")

    except Exception as e:
        print(f"⚠️  注意力可视化失败：{e}")
    finally:
        # 移除钩子，避免内存泄漏
        for hook in hooks:
            hook.remove()


# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    # 加载配置
    config = get_config()

    # 初始化YOLO模型（添加设备兼容）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo = YOLO()
    yolo.device = device
    if hasattr(yolo.model, 'to'):
        yolo.model = yolo.model.to(device)
    print(f"🔧 模型加载完成，使用设备：{device}")

    # 配置参数（兼容配置文件，增加默认值）
    dir_detect_path = config.get('dir_detect_path', 'VOCdevkit/VOC2026/images')  # 待检测图像目录
    detect_save_path = config.get('detect_save_path', 'auto_annotations')  # 标注保存目录
    vis_attention = config.get('visualize_attention', False)  # 是否可视化注意力
    conf_threshold = config.get('conf_threshold', 0.5)  # 检测置信度阈值

    # 检查输入目录
    if not os.path.exists(dir_detect_path):
        raise FileNotFoundError(f"❌ 检测目录不存在：{dir_detect_path}")

    # 获取图像列表（过滤有效格式）
    img_exts = ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')
    img_names = [f for f in os.listdir(dir_detect_path) if f.lower().endswith(img_exts)]

    if not img_names:
        print("⚠️  检测目录下无有效图像文件")
        exit(0)

    print(f"📊 开始处理 {len(img_names)} 张图像...")

    # 批量处理图像
    for img_name in tqdm(img_names, desc="自动标注进度"):
        try:
            # 1. 加载图像
            img_path = os.path.join(dir_detect_path, img_name)
            image = Image.open(img_path).convert('RGB')
            img_size = image.size  # (width, height)
            img_name_noext = os.path.splitext(img_name)[0]

            # 2. 可选：注意力可视化
            if vis_attention:
                att_save_dir = os.path.join(detect_save_path, "attention_maps", img_name_noext)
                visualize_attention(yolo, image, save_dir=att_save_dir)

            # 3. 模型检测（增加置信度过滤）
            boxes = yolo.get_box(image)
            # 过滤低置信度框（如果get_box返回包含置信度的格式，需调整）
            # 示例：如果boxes格式为 (label, ymin, xmin, ymax, xmax, conf)
            # boxes = [box for box in boxes if box[-1] >= conf_threshold]

            if not boxes:
                print(f"⚠️  {img_name} 未检测到目标，跳过标注")
                continue

            # 4. 生成YOLO格式TXT标注（替代原XML）
            generate_yolo_txt(boxes, img_size, detect_save_path, img_name_noext)

        except Exception as e:
            print(f"\n❌ 处理 {img_name} 失败：{e}")
            continue

    # 最终统计
    generated_txt = [f for f in os.listdir(detect_save_path) if f.endswith('.txt')]
    print(f"\n🎉 处理完成！")
    print(f"📈 成功生成标注：{len(generated_txt)} 个")
    print(f"💾 标注保存路径：{detect_save_path}")
    if vis_attention:
        print(f"🖼️  注意力图保存路径：{os.path.join(detect_save_path, 'attention_maps')}")