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
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

    # ---------------------------------------------------#


#   对输入图像进行resize，支持LetterBox模式
# ---------------------------------------------------#
def resize_image(image, size, letterbox_image=True):
    iw, ih = image.size
    w, h = size
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        return new_image, (nw, nh, (w - nw) // 2, (h - nh) // 2)  # 返回缩放信息
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
    # box1: [x1, y1, x2, y2]
    # box2: [x1, y1, x2, y2]
    x1min, y1min, x1max, y1max = box1
    x2min, y2min, x2max, y2max = box2

    xmin = max(x1min, x2min)
    ymin = max(y1min, y2min)
    xmax = min(x1max, x2max)
    ymax = min(y1max, y2max)

    inter_area = max(0, xmax - xmin) * max(0, ymax - ymin)
    area1 = (x1max - x1min) * (y1max - y1min)
    area2 = (x2max - x2min) * (y2max - y2min)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


# ---------------------------------------------------#
#   绘制边界框到图像上
# ---------------------------------------------------#
def draw_bbox(image, bboxes, classes, colors=None):
    image = np.array(image)
    if colors is None:
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    for box in bboxes:
        label = box[0]
        x1, y1, x2, y2 = map(int, box[1:5])
        score = box[5] if len(box) > 5 else 1.0

        color = colors[classes.index(label)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f"{label}: {score:.2f}"
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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