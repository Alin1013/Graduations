# 改造后的推理类
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont


class YOLO(object):
    _defaults = {
        "class_names": None,
        "confidence": 0.5,
        "nms_iou": 0.3,
        "device": "0"  # 0为GPU，cpu为CPU
    }

    @classmethod
    def get_defaults(cls, n):
        return cls._defaults.get(n, f"Unrecognized attribute {n}")

    def __init__(self, opt, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.weights = opt.weights
        self.imgsz = opt.shape
        self.device = opt.cuda if opt.cuda else "cpu"
        self.confidence = opt.confidence
        self.nms_iou = opt.nms_iou
        self.class_names = opt.class_names if hasattr(opt, 'class_names') else None

        # 加载模型
        self.model = YOLO(self.weights)
        if self.class_names:
            self.model.names = self.class_names

        # 初始化颜色
        self.num_classes = len(self.model.names) if self.model.names else 0
        self._init_colors()

    def _init_colors(self):
        """初始化类别颜色"""
        import colorsys
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

    def detect_image(self, image, crop=False, count=False):
        """检测图像"""
        # 转换为PIL图像
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # 预测
        results = self.model.predict(
            image,
            conf=self.confidence,
            iou=self.nms_iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )

        # 处理结果
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean([self.imgsz, self.imgsz]), 1))

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取坐标
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # 绘制边框
                draw.rectangle([x1, y1, x2, y2], outline=self.colors[cls_id], width=thickness)

                # 绘制类别和置信度
                label = f"{self.model.names[cls_id]} {conf:.2f}"
                text_size = draw.textsize(label, font)
                draw.rectangle([x1, y1 - text_size[1], x1 + text_size[0], y1], fill=self.colors[cls_id])
                draw.text([x1, y1 - text_size[1]], label, fill=(255, 255, 255), font=font)

                # 裁剪目标
                if crop:
                    self._crop_object(image, x1, y1, x2, y2, cls_id)

            # 计数
            if count:
                self._count_objects(boxes.cls.tolist())

        return image

    def _crop_object(self, image, x1, y1, x2, y2, cls_id):
        """裁剪检测到的目标"""
        import os
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        crop_img = image.crop([x1, y1, x2, y2])
        dir_save = "img_crop"
        os.makedirs(dir_save, exist_ok=True)
        crop_img.save(f"{dir_save}/crop_{cls_id}_{hash(time.time())}.png")

    def _count_objects(self, cls_list):
        """统计目标数量"""
        counts = {}
        for cls_id in cls_list:
            cls_name = self.model.names[int(cls_id)]
            counts[cls_name] = counts.get(cls_name, 0) + 1
        for name, num in counts.items():
            print(f"{name}: {num}")