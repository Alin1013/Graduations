from random import sample, shuffle
import os
import cv2
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, epoch_length, mosaic, train, mosaic_ratio=0.7):
        super(YoloDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape  # (h, w)
        self.num_classes = num_classes
        self.epoch_length = epoch_length
        self.mosaic = mosaic
        self.train = train
        self.mosaic_ratio = mosaic_ratio

        self.epoch_now = -1
        self.length = len(self.annotation_lines)
        # 过滤无效标注行
        self._filter_invalid_annotations()

    def _filter_invalid_annotations(self):
        """过滤无效的标注行（图像路径不存在或格式错误）"""
        valid_lines = []
        for line in self.annotation_lines:
            try:
                img_path = line.split()[0]
                if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                    valid_lines.append(line)
                else:
                    print(f"警告：无效图像路径或空文件 - {img_path}，已过滤")
            except:
                print(f"警告：标注行格式错误 - {line}，已过滤")
        self.annotation_lines = valid_lines
        self.length = len(self.annotation_lines)
        if self.length == 0:
            raise ValueError("错误：没有有效的标注数据，请检查标注文件")

    def set_epoch(self, epoch):
        """更新当前epoch，用于控制Mosaic增强的启用阶段"""
        self.epoch_now = epoch

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length

        if self.mosaic and self.train:  # 仅训练模式启用Mosaic
            # 控制Mosaic在训练前期启用，后期逐渐减少
            if self.rand() < 0.5 and self.epoch_now < self.epoch_length * self.mosaic_ratio:
                # 确保有足够的样本进行Mosaic增强
                if self.length >= 4:
                    lines = sample(self.annotation_lines, 3)
                    lines.append(self.annotation_lines[index])
                    shuffle(lines)
                    image, box = self.get_random_data_with_Mosaic(lines, self.input_shape)
                else:
                    # 样本不足时使用普通增强
                    image, box = self.get_random_data(self.annotation_lines[index], self.input_shape, random=self.train)
            else:
                image, box = self.get_random_data(self.annotation_lines[index], self.input_shape, random=self.train)
        else:
            image, box = self.get_random_data(self.annotation_lines[index], self.input_shape, random=self.train)

        # 图像预处理：归一化并转换通道顺序 (HWC -> CHW)
        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box = np.array(box, dtype=np.float32)

        # 标注框归一化：转换为中心点+宽高格式
        if len(box) != 0:
            h, w = self.input_shape
            # 坐标归一化到[0,1]
            box[:, [0, 2]] /= w
            box[:, [1, 3]] /= h
            # 转换为 (中心点x, 中心点y, 宽, 高)
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]  # 宽高 = 右下角 - 左上角
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2  # 中心点 = 左上角 + 宽高/2

        return image, box

    def rand(self, a=0, b=1):
        """生成[a, b)区间的随机数"""
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=0.3, hue=0.1, sat=0.7, val=0.4, random=True):
        """基础数据增强：缩放、裁剪、翻转、色域变换"""
        line = annotation_line.split()
        img_path = line[0]

        # 读取图像（添加异常处理）
        try:
            image = Image.open(img_path)
            image = cvtColor(image)  # 转换为RGB
        except (UnidentifiedImageError, IOError) as e:
            print(f"警告：无法读取图像 {img_path}，错误：{e}，使用随机图像替代")
            # 生成随机图像作为替代
            h, w = input_shape
            image = Image.new('RGB', (w, h), (128, 128, 128))

        iw, ih = image.size
        h, w = input_shape

        # 解析标注框 (x1, y1, x2, y2, cls)
        box = []
        if len(line) > 1:
            try:
                box = np.array([np.array(list(map(int, b.split(',')))) for b in line[1:]])
            except:
                print(f"警告：标注格式错误 - {annotation_line}，忽略标注框")
                box = np.array([])

        if not random:
            # 验证模式：保持比例缩放，边缘补灰条
            scale = min(w / iw, h / ih)
            nw, nh = int(iw * scale), int(ih * scale)
            dx, dy = (w - nw) // 2, (h - nh) // 2

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # 调整标注框
            if len(box) > 0:
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                # 限制坐标在有效范围内
                box[:, [0, 2]] = np.clip(box[:, [0, 2]], 0, w)
                box[:, [1, 3]] = np.clip(box[:, [1, 3]], 0, h)
                # 过滤无效框（宽高>1）
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]

            return image_data, box

        # 训练模式：随机增强
        # 随机缩放和扭曲比例
        new_ar = (iw / ih) * (self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter))
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # 随机放置图像，边缘补灰条
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # 随机水平翻转
        flip = self.rand() < 0.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 转换为numpy数组
        image_data = np.array(image, np.uint8)

        # HSV色域随机变换
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue_channel, sat_channel, val_channel = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype

        # 应用变换（使用LUT加速）
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge([
            cv2.LUT(hue_channel, lut_hue),
            cv2.LUT(sat_channel, lut_sat),
            cv2.LUT(val_channel, lut_val)
        ])
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        # 调整标注框
        if len(box) > 0:
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            # 翻转时调整x坐标
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            # 限制坐标范围
            box[:, [0, 2]] = np.clip(box[:, [0, 2]], 0, w)
            box[:, [1, 3]] = np.clip(box[:, [1, 3]], 0, h)
            # 过滤无效框
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, box

    def merge_bboxes(self, bboxes, cutx, cuty):
        """合并Mosaic增强中的标注框，处理跨区域的框"""
        merge_bbox = []
        for i, box_group in enumerate(bboxes):
            for box in box_group:
                if len(box) < 5:  # 确保标注框格式正确 (x1, y1, x2, y2, cls)
                    continue
                x1, y1, x2, y2, cls = box[:5]

                # 根据图像位置裁剪框
                if i == 0:  # 左上区域
                    if y1 > cuty or x1 > cutx:
                        continue
                    x2 = min(x2, cutx)
                    y2 = min(y2, cuty)
                elif i == 1:  # 左下区域
                    if y2 < cuty or x1 > cutx:
                        continue
                    x2 = min(x2, cutx)
                    y1 = max(y1, cuty)
                elif i == 2:  # 右下区域
                    if y2 < cuty or x2 < cutx:
                        continue
                    x1 = max(x1, cutx)
                    y1 = max(y1, cuty)
                elif i == 3:  # 右上区域
                    if y1 > cuty or x2 < cutx:
                        continue
                    x1 = max(x1, cutx)
                    y2 = min(y2, cuty)

                # 过滤裁剪后无效的框
                if x2 - x1 > 1 and y2 - y1 > 1:
                    merge_bbox.append([x1, y1, x2, y2, cls])
        return merge_bbox

    def get_random_data_with_Mosaic(self, annotation_lines, input_shape, jitter=0.3, hue=0.1, sat=0.7, val=0.4):
        """Mosaic数据增强：将4张图像按象限拼接"""
        h, w = input_shape
        # 随机选择拼接中心点（避免过于边缘）
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)
        cutx, cuty = int(w * min_offset_x), int(h * min_offset_y)

        image_datas = []
        box_datas = []

        for idx, line in enumerate(annotation_lines):
            line_content = line.split()
            img_path = line_content[0] if line_content else ""

            # 读取图像（添加异常处理）
            try:
                image = Image.open(img_path)
                image = cvtColor(image)
            except (UnidentifiedImageError, IOError) as e:
                print(f"警告：Mosaic增强中无法读取图像 {img_path}，错误：{e}，使用空白图像替代")
                image = Image.new('RGB', (w, h), (128, 128, 128))

            iw, ih = image.size
            # 解析标注框
            box = []
            if len(line_content) > 1:
                try:
                    box = np.array([np.array(list(map(int, b.split(',')))) for b in line_content[1:]])
                except:
                    print(f"警告：Mosaic中标注格式错误 - {line}，忽略标注框")
                    box = np.array([])

            # 随机翻转
            flip = self.rand() < 0.5
            if flip and len(box) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = iw - box[:, [2, 0]]  # 翻转x坐标

            # 随机缩放和扭曲
            new_ar = (iw / ih) * (self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter))
            scale = self.rand(0.4, 1)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            # 计算图像放置位置（修正偏移量计算）
            if idx == 0:  # 左上
                dx = cutx - nw
                dy = cuty - nh
            elif idx == 1:  # 左下
                dx = cutx - nw
                dy = cuty
            elif idx == 2:  # 右下
                dx = cutx
                dy = cuty
            else:  # 右上
                dx = cutx
                dy = cuty - nh

            # 放置图像到画布
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_datas.append(np.array(new_image))

            # 调整标注框
            box_data = []
            if len(box) > 0:
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                # 限制坐标范围
                box[:, [0, 2]] = np.clip(box[:, [0, 2]], 0, w)
                box[:, [1, 3]] = np.clip(box[:, [1, 3]], 0, h)
                # 过滤无效框
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                valid_mask = np.logical_and(box_w > 1, box_h > 1)
                box_data = box[valid_mask]

            box_datas.append(box_data)

        # 拼接4张图像（优化内存操作）
        new_image = np.zeros((h, w, 3), dtype=np.uint8)
        new_image[:cuty, :cutx] = image_datas[0][:cuty, :cutx]  # 左上
        new_image[cuty:, :cutx] = image_datas[1][cuty:, :cutx]  # 左下
        new_image[cuty:, cutx:] = image_datas[2][cuty:, cutx:]  # 右下
        new_image[:cuty, cutx:] = image_datas[3][:cuty, cutx:]  # 右上

        # HSV色域变换
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue_channel, sat_channel, val_channel = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype = new_image.dtype

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        new_image = cv2.merge([
            cv2.LUT(hue_channel, lut_hue),
            cv2.LUT(sat_channel, lut_sat),
            cv2.LUT(val_channel, lut_val)
        ])
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

        # 合并标注框
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)
        return new_image, new_boxes


def yolo_dataset_collate(batch):
    """DataLoader的批处理函数：处理不同长度的标注框"""
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    # 图像转换为Tensor
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    # 标注框转换为Tensor（保留批次内的不同长度）
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    return images, bboxes