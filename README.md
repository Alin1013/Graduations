手势检测项目（基于 YOLOv8）
项目简介
本项目是一个基于 YOLOv8 的手势检测系统，能够实时实现对相关手势的实时检测与识别。项目采用 VOC 格式数据集进行训练，提供了完整的模型训练、评估和部署流程，并通过 Streamlit 构建了交互式 Web 应用界面，支持图像、视频和摄像头实时检测。
项目结构
plaintext
Graduation_Project/
├── VOCdevkit/                  # VOC格式数据集
│   └── VOC2026/
│       ├── Annotations/        # 标签文件（XML格式）
│       ├── ImageSets/Main/     # 训练索引文件
│       └── JPEGImages/         # 图像文件
├── nets/                       # 网络模型定义
│   ├── yolov8.py               # YOLOv8模型
│   ├── CSPdarknet53_tiny.py    # 轻量化主干网络
│   └── CSPdarknet.py           # CSPDarknet53主干网络
├── model_data/                 # 模型配置文件
│   └── gesture.yaml            # 数据集与类别配置
├── runs/                       # 训练结果
│   └── detect/                 # 检测任务相关结果
├── utils/                      # 工具函数
├── 2026_train.txt              # 训练集路径与标签
├── 2026_val.txt                # 验证集路径与标签
├── summary.py                  # 网络结构查看工具
├── train.py                    # 模型训练脚本
├── predict.py                  # 模型预测脚本
├── get_map.py                  # 模型评估（计算mAP）
├── gen_annotation.py           # 生成标注文件工具
├── kmeans_for_anchors.py       # 锚框聚类工具
├── gesture_streamlit.py        # Streamlit Web应用
├── requirements.txt            # 项目依赖
└── README.md                   # 项目说明文档

环境配置
# 安装依赖包
pip install -r requirements.txt


数据集准备
项目使用 VOC 格式数据集，结构如下：
图像文件：VOCdevkit/VOC2026/JPEGImages/
标注文件：VOCdevkit/VOC2026/Annotations/（XML 格式）
训练 / 验证集划分：VOCdevkit/VOC2026/ImageSets/Main/

数据集配置
修改model_data/gesture.yaml文件配置数据集路径和类别：
yaml
# 数据集路径
train: /path/to/train/images
val: /path/to/val/images

# 类别信息（8种手势）
nc: 8
names: ["up","down","left","right","front","back","clockwise","anticlockwise"]

# 检测参数
confidence: 0.5
nms_iou: 0.3

模型训练
# GPU训练
python train.py --epochs 100 --imgsz 640 --device 0 --batch-size 8 --weights ./yolov8n.pt

#CPU训练（无GPU时）
# python3 train.py --epochs 10 --imgsz 640 --device cpu --batch-size 4 --weights ./yolov8n.pt

# 冻结训练
# python3 train.py --epochs 100 --freeze --freeze-epochs 50 --device cpu

#预测命令
# python3 predict.py --tiny --cuda \
#    --weights runs/detect/train/weights/best.pt \
#    --shape 640


训练参数说明：
--epochs：训练轮数
--imgsz：输入图像尺寸（默认 640）
--device：训练设备（cpu 或 GPU 编号）
--batch-size：批次大小
--weights：预训练权重路径
--freeze：启用冻结训练

模型评估
使用get_map.py计算模型的 mAP（平均精度均值）：

# 计算VOC标准mAP
python get_map.py --weights runs/detect/train/weights/best.pt --mode 0

# 仅计算mAP（需先生成预测结果和真实框）
python get_map.py --mode 3

#通过 Streamlit 启动交互式 Web 应用：
streamlit run gesture_streamlit.py

python summary.py
该脚本会输出 YOLOv8 整体网络结构以及 CSPDarknet53 主干网络的详细信息。

python kmeans_for_anchors.py


基于Ultralytics YOLOv8框架开发
数据集格式遵循 PASCAL VOC 标准
Web 应用使用 Streamlit 构建