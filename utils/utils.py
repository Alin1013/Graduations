import numpy as np
import os
import random
import torch
from PIL import Image
import cv2
from torch.nn import functional as F


# ---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错
# ---------------------------------------------------------#
def cvtColor(image):
    """将图像转换成RGB图像，防止灰度图在预测时报错"""
    # 处理PIL图像
    if isinstance(image, Image.Image):
        if image.mode != 'RGB':
            return image.convert('RGB')
        return image
    # 处理numpy数组（OpenCV格式）
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV默认BGR
        elif len(image.shape) == 2:  # 灰度图
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    raise TypeError(f"不支持的图像类型: {type(image)}")

    # ---------------------------------------------------#


#   对输入图像进行resize，支持LetterBox模式
# ---------------------------------------------------#
def resize_image(image, size, letterbox_image=True):
    """对输入图像进行resize，支持LetterBox模式"""
    if not isinstance(image, Image.Image):
        raise TypeError("image必须是PIL.Image对象")
    if not isinstance(size, (tuple, list)) or len(size) != 2:
        raise ValueError("size必须是包含两个元素的元组或列表 (width, height)")

    iw, ih = image.size
    w, h = size
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(round(iw * scale))  # 使用round确保整数精度
        nh = int(round(ih * scale))

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        # 返回缩放后的图像和缩放信息（新宽、新高、x偏移、y偏移）
        return new_image, (nw, nh, (w - nw) // 2, (h - nh) // 2)
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
        return new_image, (w, h, 0, 0)


# ---------------------------------------------------#
#   从文件读取类别名称
# ---------------------------------------------------#
def get_classes(classes_path):
    if not os.path.exists(classes_path):
        raise FileNotFoundError(f"类别文件不存在: {classes_path}")
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


# ---------------------------------------------------#
#   从文件读取先验框
# ---------------------------------------------------#
def get_anchors(anchors_path):
    if not os.path.exists(anchors_path):
        raise FileNotFoundError(f"先验框文件不存在: {anchors_path}")
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)


# ---------------------------------------------------#
#   获取当前学习率
# ---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# ---------------------------------------------------#
#   图像预处理（归一化）
# ---------------------------------------------------#
def preprocess_input(image):
    image = np.array(image, dtype=np.float32)
    image /= 255.0
    return image


# ---------------------------------------------------#
#   设置随机种子，保证实验可复现
# ---------------------------------------------------#
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------#
#   计算两个矩形框的交并比(IoU)
# ---------------------------------------------------#
def bbox_iou(box1, box2):
    """
    计算边界框的交并比（IoU）
    支持单个框: box1=[x1,y1,x2,y2], box2=[x1,y1,x2,y2]
    支持批量框: box1=(N,4), box2=(M,4)
    """
    # 转换为numpy数组
    box1 = np.array(box1, dtype=np.float32).reshape(-1, 4)
    box2 = np.array(box2, dtype=np.float32).reshape(-1, 4)

    x1min, y1min, x1max, y1max = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    x2min, y2min, x2max, y2max = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # 计算交集
    xmin = np.maximum(x1min[:, np.newaxis], x2min)
    ymin = np.maximum(y1min[:, np.newaxis], y2min)
    xmax = np.minimum(x1max[:, np.newaxis], x2max)
    ymax = np.minimum(y1max[:, np.newaxis], y2max)

    inter_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # 计算面积
    area1 = (x1max - x1min) * (y1max - y1min)
    area2 = (x2max - x2min) * (y2max - y2min)
    union_area = area1[:, np.newaxis] + area2 - inter_area

    # 计算IoU
    iou = inter_area / (union_area + 1e-8)  # 避免除零

    # 如果是单个框，返回标量
    return iou[0, 0] if iou.size == 1 else iou


# ---------------------------------------------------#
#   绘制边界框到图像上
# ---------------------------------------------------#
def draw_bbox(image, bboxes, classes, colors=None):
    """在图像上绘制边界框和类别信息"""
    # 转换为numpy数组（OpenCV格式）
    image = np.array(image)
    # 创建类别到索引的映射，提高查询效率
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    if colors is None:
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    for box in bboxes:
        try:
            label = box[0]
            x1, y1, x2, y2 = map(int, box[1:5])
            score = box[5] if len(box) > 5 else 1.0

            # 获取类别颜色
            cls_idx = class_to_idx.get(label, 0)  # 未知类别使用第0种颜色
            color = colors[cls_idx]

            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # 绘制标签（支持中文）
            text = f"{label}: {score:.2f}"
            # 确保文本在图像范围内
            y_text = max(y1 - 10, 10)
            # 使用支持中文的字体（需要提前准备字体文件）
            try:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, text, (x1, y_text), font, 0.5, color, 2)
            except:
                # 回退方案
                cv2.putText(image, text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except Exception as e:
            print(f"绘制边界框失败: {e}, 跳过该框")
            continue

    return Image.fromarray(image)


# ---------------------------------------------------#
#   图像增强：随机水平翻转
# ---------------------------------------------------#
def random_flip(image, bboxes, prob=0.5):
    if random.random() < prob:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        w = image.size[0]
        new_bboxes = []
        for box in bboxes:
            label, x1, y1, x2, y2 = box
            new_x1 = w - x2
            new_x2 = w - x1
            new_bboxes.append([label, new_x1, y1, new_x2, y2])
        return image, new_bboxes
    return image, bboxes