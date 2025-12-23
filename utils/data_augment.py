import cv2
import numpy as np
import random


class GestureAugmentor:
    """手势数据增强工具类，增加样本多样性"""

    def __init__(self):
        self.transforms = [
            self.random_rotate,
            self.random_scale,
            self.random_shift,
            self.random_flip,
            self.random_brightness_contrast,
            self.add_noise,
            self.random_crop
        ]

    def random_rotate(self, img, bbox):
        """随机旋转(-30°~30°)"""
        h, w = img.shape[:2]
        angle = random.uniform(-30, 30)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        img_rot = cv2.warpAffine(img, M, (w, h))

        # 调整边界框
        x1, y1, x2, y2 = bbox
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        pts_rot = cv2.transform(np.array([pts]), M)[0]
        x1, y1 = np.min(pts_rot[:, 0]), np.min(pts_rot[:, 1])
        x2, y2 = np.max(pts_rot[:, 0]), np.max(pts_rot[:, 1])
        return img_rot, [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]

    def random_scale(self, img, bbox):
        """随机缩放(0.8~1.2倍)"""
        scale = random.uniform(0.8, 1.2)
        h, w = img.shape[:2]
        img_scaled = cv2.resize(img, (int(w * scale), int(h * scale)))

        # 调整边界框
        x1, y1, x2, y2 = bbox
        return img_scaled, [x1 * scale, y1 * scale, x2 * scale, y2 * scale]

    def random_shift(self, img, bbox, max_shift=0.1):
        """随机平移"""
        h, w = img.shape[:2]
        dx = random.uniform(-w * max_shift, w * max_shift)
        dy = random.uniform(-h * max_shift, h * max_shift)

        M = np.float32([[1, 0, dx], [0, 1, dy]])
        img_shifted = cv2.warpAffine(img, M, (w, h))

        # 调整边界框
        x1, y1, x2, y2 = bbox
        return img_shifted, [x1 + dx, y1 + dy, x2 + dx, y2 + dy]

    def random_flip(self, img, bbox):
        """随机水平翻转"""
        if random.random() < 0.5:
            img_flip = cv2.flip(img, 1)
            h, w = img.shape[:2]
            x1, y1, x2, y2 = bbox
            return img_flip, [w - x2, y1, w - x1, y2]
        return img, bbox

    def random_brightness_contrast(self, img, bbox):
        """随机调整亮度对比度"""
        alpha = random.uniform(0.5, 1.5)  # 对比度
        beta = random.uniform(-30, 30)  # 亮度
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        return img, bbox

    def add_noise(self, img, bbox):
        """添加高斯噪声"""
        if random.random() < 0.3:
            h, w = img.shape[:2]
            noise = np.random.normal(0, 15, (h, w, 3)).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img, bbox

    def random_crop(self, img, bbox):
        """随机裁剪手势区域"""
        if random.random() < 0.5:
            h, w = img.shape[:2]
            x1, y1, x2, y2 = bbox
            # 扩大裁剪区域
            crop_x1 = max(0, int(x1 - random.uniform(0, 0.2 * (x2 - x1))))
            crop_y1 = max(0, int(y1 - random.uniform(0, 0.2 * (y2 - y1))))
            crop_x2 = min(w, int(x2 + random.uniform(0, 0.2 * (x2 - x1))))
            crop_y2 = min(h, int(y2 + random.uniform(0, 0.2 * (y2 - y1))))

            img_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
            # 调整边界框相对位置
            new_bbox = [
                x1 - crop_x1, y1 - crop_y1,
                x2 - crop_x1, y2 - crop_y1
            ]
            return img_crop, new_bbox
        return img, bbox

    def __call__(self, img, bbox):
        """应用随机增强组合"""
        augmented_img, augmented_bbox = img.copy(), bbox.copy()
        # 随机选择3-5种增强方式
        num_transforms = random.randint(3, 5)
        selected_transforms = random.sample(self.transforms, num_transforms)

        for transform in selected_transforms:
            augmented_img, augmented_bbox = transform(augmented_img, augmented_bbox)

        return augmented_img, augmented_bbox