from ultralytics import YOLO

class YOLOv8Body:
    def __init__(self, model_type="yolov8n.pt", num_classes=None, pretrained=True):
        """
        model_type: 模型类型，可选yolov8n/s/m/l/x.pt(预训练)或自定义路径
        num_classes: 类别数，None则使用预训练模型的类别数
        """
        self.model = YOLO(model_type)
        if num_classes and num_classes != self.model.model.yaml['nc']:
            # 重新初始化头部以适应新的类别数
            self.model.model.nc = num_classes
            self.model.model.names = [f"class_{i}" for i in range(num_classes)]
            self.model.model.reset_weights(self.model.model.head.names)

    def train(self, data, epochs=100, imgsz=640, **kwargs):
        """训练接口"""
        return self.model.train(data=data, epochs=epochs, imgsz=imgsz,** kwargs)

    def predict(self, source, **kwargs):
        """预测接口"""
        return self.model.predict(source=source,** kwargs)

    def val(self, data, **kwargs):
        """验证接口"""
        return self.model.val(data=data,** kwargs)

    def save(self, path):
        """保存模型"""
        self.model.save(path)