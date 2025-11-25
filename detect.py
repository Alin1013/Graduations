# -----------------------------------------------------------------------#
#   detect.py 是用来尝试利用小模型半自动化进行标注数据
# -----------------------------------------------------------------------#
import numpy as np
from PIL import Image
from get_yaml import get_config
import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from yolo import YOLO
from gen_annotation import GEN_Annotations


def visualize_attention(yolo_model, image, save_dir="attention_maps"):
    """
    可视化CBAM注意力模块的输出
    :param yolo_model: YOLO模型实例
    :param image: PIL图像对象
    :param save_dir: 注意力图保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    attention_maps = []

    # 定义钩子函数获取注意力图
    def hook_fn(module, input, output):
        attention_maps.append(output)

    # 注册钩子（适配YOLOv4的PANet结构，根据实际模型调整模块路径）
    # 这里假设CBAM模块在neck的PANet blocks中
    try:
        # 尝试获取PANet中的CBAM模块（根据实际模型结构调整）
        if hasattr(yolo_model.model.neck, 'pan_blocks'):
            # 为每个PAN块的CBAM注册钩子
            for i, block in enumerate(yolo_model.model.neck.pan_blocks):
                if hasattr(block, 'cbam'):
                    block.cbam.register_forward_hook(hook_fn)
        else:
            print("未找到PANet模块，无法可视化注意力图")
            return
    except AttributeError as e:
        print(f"注册注意力钩子失败: {e}")
        return

    # 图像预处理（与模型输入要求一致）
    image_np = np.array(image)
    image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float() / 255.0  # 归一化
    image_tensor = image_tensor.unsqueeze(0)  # 添加批次维度
    if yolo_model.device != "cpu":
        image_tensor = image_tensor.cuda()

    # 前向传播触发钩子
    with torch.no_grad():
        yolo_model.model(image_tensor)

    # 保存注意力图
    for idx, am in enumerate(attention_maps):
        # 处理注意力图（通道平均，缩放到原图尺寸）
        am_np = am.squeeze().mean(dim=0).cpu().detach().numpy()
        am_resized = np.array(Image.fromarray(am_np).resize(image.size[::-1], Image.BILINEAR))

        # 保存热力图
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.imshow(am_resized, cmap='jet', alpha=0.5)  # 叠加注意力图
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f"attention_{idx}.png"), bbox_inches='tight', pad_inches=0)
        plt.close()

    print(f"注意力图已保存至 {save_dir}")


if __name__ == "__main__":
    # 加载配置
    config = get_config()
    yolo = YOLO()  # 初始化YOLO模型

    dir_detect_path = config['dir_detect_path']
    detect_save_path = config['detect_save_path']
    vis_attention = config.get('visualize_attention', False)  # 从配置文件控制是否可视化

    img_names = os.listdir(dir_detect_path)
    for img_name in tqdm(img_names):
        # 处理图像文件
        if img_name.lower().endswith(
                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path = os.path.join(dir_detect_path, img_name)
            image = Image.open(image_path)

            # 可视化注意力图（可选）
            if vis_attention:
                visualize_attention(yolo, image,
                                    save_dir=os.path.join(detect_save_path, "attention_maps", img_name.split('.')[0]))

            # 获取检测框并生成标注
            boxes = yolo.get_box(image)
            if not os.path.exists(detect_save_path):
                os.makedirs(detect_save_path)

            # 生成标注文件
            annotation = GEN_Annotations(img_name)
            w, h = np.array(image.size[::-1])  # 注意PIL的size是(w, h)，这里转换为(h, w)
            annotation.set_size(w, h, 3)
            if boxes:
                for box in boxes:
                    label, ymin, xmin, ymax, xmax = box
                    annotation.add_pic_attr(label, xmin, ymin, xmax, ymax)
                annotation_path = os.path.join(detect_save_path, img_name.split('.')[0])
                annotation.savefile(f"{annotation_path}.xml")