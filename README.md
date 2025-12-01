# 手势检测项目（基于 YOLOv8）

# 项目简介
本项目是一个基于 YOLOv8 的手势检测系统，能够实时实现对相关手势的实时检测与识别。项目采用 VOC 格式数据集进行训练，提供了完整的模型训练、评估和部署流程，并通过 Streamlit 构建了交互式 Web 应用界面，支持图像、视频和摄像头实时检测。

# 环境配置
#安装依赖包
pip install -r requirements.txt


# 数据集准备
项目使用 VOC 格式数据集，结构如下：
图像文件：VOCdevkit/VOC2026/JPEGImages/
标注文件：VOCdevkit/VOC2026/Annotations/（XML 格式）
训练 / 验证集划分：VOCdevkit/VOC2026/ImageSets/Main/

# 模型训练
# GPU训练
python train.py --epochs 100 --imgsz 640 --device 0 --batch-size 8 --weights ./yolov8n.pt

# CPU训练（无GPU时）
python3 train.py --epochs 10 --imgsz 640 --device cpu --batch-size 4 --weights ./yolov8n.pt

# 冻结训练
 python3 train.py --epochs 100 --freeze --freeze-epochs 50 --device cpu


# 训练参数说明：
--epochs：训练轮数
--imgsz：输入图像尺寸（默认 640）
--device：训练设备（cpu 或 GPU 编号）
--batch-size：批次大小
--weights：预训练权重路径
--freeze：启用冻结训练

# 通过 Streamlit 启动交互式 Web 应用：
streamlit run gesture_streamlit.py


