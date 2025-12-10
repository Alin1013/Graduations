import cv2
import time
import numpy as np
import os
import colorsys
from ultralytics import YOLO as UltralyticsYOLO  # 重命名避免冲突
from PIL import Image, ImageDraw, ImageFont


class YOLO(object):
    """
    YOLOv8推理类
    支持图像检测、目标裁剪、数量统计
    """
    _defaults = {
        "class_names": None,
        "confidence": 0.5,
        "nms_iou": 0.3,
        "device": "cpu"  # 默认CPU，"0"为GPU
    }

    @classmethod
    def get_defaults(cls, n):
        """获取默认配置"""
        return cls._defaults.get(n, f"Unrecognized attribute {n}")

    def __init__(self, opt, **kwargs):
        """
        初始化YOLO模型
        :param opt: 命令行参数对象
        :param kwargs: 额外配置参数
        """
        # 初始化默认配置
        self.__dict__.update(self._defaults)

        # 更新额外配置
        for name, value in kwargs.items():
            setattr(self, name, value)

        # 从opt解析参数（核心修复）
        self.weights = getattr(opt, 'weights', 'yolov8n.pt')
        self.imgsz = getattr(opt, 'shape', 640)
        self.confidence = getattr(opt, 'confidence', 0.5)
        self.nms_iou = getattr(opt, 'nms_iou', 0.3)
        self.class_names = getattr(opt, 'class_names', None)

        # 设备配置（修复GPU/CPU逻辑）
        if hasattr(opt, 'cuda') and opt.cuda:
            self.device = "0"  # GPU
        else:
            self.device = "cpu"  # CPU

        # 加载模型（修复命名冲突）
        try:
            self.model = UltralyticsYOLO(self.weights)
            print(f"✅ 模型加载成功: {self.weights}")
        except Exception as e:
            raise RuntimeError(f"❌ 模型加载失败: {e}")

        # 更新类别名称
        if self.class_names:
            self.model.names = self.class_names
            print(f"✅ 类别名称已更新: {self.class_names}")

        # 初始化类别颜色
        self.num_classes = len(self.model.names) if self.model.names else 0
        if self.num_classes > 0:
            self._init_colors()
        else:
            self.colors = [(255, 0, 0)]  # 默认红色
            print("⚠️  未检测到类别名称，使用默认颜色")

    def _init_colors(self):
        """初始化类别颜色（HSV色域均匀分布）"""
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = []
        for hsv in hsv_tuples:
            rgb = colorsys.hsv_to_rgb(*hsv)
            rgb = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
            self.colors.append(rgb)

    def _get_font(self, image):
        """获取适配图像大小的字体"""
        try:
            # 优先使用中文字体
            font_size = np.floor(3e-2 * image.size[1] + 0.5).astype('int32')
            return ImageFont.truetype(font='model_data/simhei.ttf', size=font_size)
        except:
            # 回退到默认字体
            return ImageFont.load_default()

    def _crop_object(self, image, x1, y1, x2, y2, cls_id):
        """
        裁剪检测到的目标
        :param image: PIL图像对象
        :param x1, y1, x2, y2: 目标坐标
        :param cls_id: 类别ID
        """
        try:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            # 确保坐标在图像范围内
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.size[0], x2)
            y2 = min(image.size[1], y2)

            crop_img = image.crop([x1, y1, x2, y2])
            dir_save = "img_crop"
            os.makedirs(dir_save, exist_ok=True)

            # 生成唯一文件名
            cls_name = self.model.names[cls_id] if cls_id < len(self.model.names) else f"cls_{cls_id}"
            filename = f"{dir_save}/crop_{cls_name}_{int(time.time() * 1000)}.png"
            crop_img.save(filename)
            print(f"📸 裁剪目标保存: {filename}")
        except Exception as e:
            print(f"⚠️  裁剪目标失败: {e}")

    def _count_objects(self, cls_list):
        """
        统计目标数量
        :param cls_list: 类别ID列表
        :return: 统计字典
        """
        counts = {}
        for cls_id in cls_list:
            cls_id = int(cls_id)
            if cls_id < len(self.model.names):
                cls_name = self.model.names[cls_id]
            else:
                cls_name = f"未知类别_{cls_id}"
            counts[cls_name] = counts.get(cls_name, 0) + 1

        # 打印统计结果
        print("\n📊 目标统计:")
        for name, num in counts.items():
            print(f"  {name}: {num}")
        return counts

    def detect_image(self, image, crop=False, count=False):
        """
        检测图像并绘制结果
        :param image: PIL图像/np.ndarray
        :param crop: 是否裁剪目标
        :param count: 是否统计数量
        :return: 绘制后的PIL图像
        """
        # 格式转换：np.ndarray → PIL Image
        if isinstance(image, np.ndarray):
            # 处理BGR格式（OpenCV）
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise TypeError(f"❌ 不支持的图像类型: {type(image)}")

        # 模型预测
        try:
            results = self.model.predict(
                image,
                conf=self.confidence,
                iou=self.nms_iou,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
                show_labels=False,
                show_conf=False
            )
        except Exception as e:
            raise RuntimeError(f"❌ 预测失败: {e}")

        # 初始化绘制工具
        draw = ImageDraw.Draw(image)
        font = self._get_font(image)
        thickness = int(max((image.size[0] + image.size[1]) // np.mean([self.imgsz, self.imgsz]), 1))

        # 收集类别ID用于统计
        cls_list = []

        # 处理检测结果
        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            for box in boxes:
                # 获取基本信息
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_list.append(cls_id)

                # 选择颜色（处理类别ID超出范围的情况）
                color = self.colors[cls_id] if cls_id < len(self.colors) else (255, 0, 0)

                # 绘制边框
                draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)

                # 绘制类别和置信度标签
                cls_name = self.model.names[cls_id] if cls_id < len(self.model.names) else f"cls_{cls_id}"
                label = f"{cls_name} {conf:.2f}"

                # 计算标签背景大小
                text_size = draw.textsize(label, font)
                text_x = x1
                text_y = y1 - text_size[1] if y1 - text_size[1] > 0 else y1 + thickness

                # 绘制标签背景
                draw.rectangle(
                    [text_x, text_y, text_x + text_size[0], text_y + text_size[1]],
                    fill=color
                )
                # 绘制标签文字
                draw.text([text_x, text_y], label, fill=(255, 255, 255), font=font)

                # 裁剪目标
                if crop:
                    self._crop_object(image, x1, y1, x2, y2, cls_id)

        # 统计目标数量
        if count and cls_list:
            self._count_objects(cls_list)

        return image

    def get_FPS(self, image, test_interval=100):
        """
        计算FPS
        :param image: 测试图像
        :param test_interval: 测试次数
        :return: 单帧平均耗时
        """
        if not isinstance(image, Image.Image):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                raise TypeError(f"❌ 不支持的图像类型: {type(image)}")

        # 预热模型
        print("🔥 预热模型...")
        for _ in range(10):
            self.detect_image(image, crop=False, count=False)

        # 正式测试
        print(f"⏱️  开始FPS测试（{test_interval}次）...")
        start_time = time.time()
        for _ in range(test_interval):
            self.detect_image(image, crop=False, count=False)
        end_time = time.time()

        # 计算结果
        tact_time = (end_time - start_time) / test_interval
        fps = 1 / tact_time
        print(f"\n📊 FPS测试结果:")
        print(f"   单帧耗时: {tact_time:.4f} 秒")
        print(f"   FPS: {fps:.2f} (batch_size=1)")

        return tact_time