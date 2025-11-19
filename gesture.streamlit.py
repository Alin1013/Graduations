"""Create an Object Detection Web App using PyTorch and Streamlit."""
# import libraries
from PIL import Image
from torchvision import models, transforms
import torch
import streamlit as st
# 替换 YOLOv4 导入为 YOLOv8
from ultralytics import YOLO
import os
import urllib
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
# 设置网页的icon
st.set_page_config(page_title='Gesture Detector', page_icon='✌',
                   layout='centered', initial_sidebar_state='expanded')

RTC_CONFIGURATION = RTCConfiguration(
    {
      "RTCIceServers": [{  # 修正参数名：RTCIceServer → RTCIceServers
        "urls": ["stun:stun.l.google.com:19302"],
        "username": "pikachu",
        "credential": "1234",
      }]
    }
)

def main():
    # Render the readme as markdown using st.markdown.
    # 增加文件存在性判断，避免报错
    if os.path.exists("instructions.md"):
        readme_text = st.markdown(open("instructions.md", encoding='utf-8').read())
    else:
        readme_text = st.markdown("# 手势检测Web应用\n请上传图片/视频或使用摄像头进行检测")

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "Run the app", "Show the source code"])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "Show the source code":
        readme_text.empty()
        # 修正文件名为当前脚本名
        if os.path.exists(__file__):
            st.code(open(__file__, encoding='utf-8').read())
        else:
            st.warning("Source code file not found")
    elif app_mode == "Run the app":
        # Download external dependencies.
        for filename in EXTERNAL_DEPENDENCIES.keys():
            download_file(filename)

        readme_text.empty()
        run_the_app()

# -------------------------- 关键修改：YOLOv8 官方权重配置 --------------------------
# YOLOv8 官方权重下载地址（ultralytics官方CDN，稳定下载）
# 权重说明：n(纳米) < s(小) < m(中) < l(大) < x(超大)，尺寸越大精度越高、速度越慢
EXTERNAL_DEPENDENCIES = {
    "yolov8n.pt": {  # 纳米模型（最快，适合CPU/边缘设备）
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        "size": 6257408  # 文件大小：~6.26 MB
    },
    "yolov8s.pt": {  # 小型模型（平衡速度与精度）
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt",
        "size": 24555520  # 文件大小：~24.56 MB
    },
    "yolov8m.pt": {  # 中型模型（高精度）
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt",
        "size": 56834560  # 文件大小：~56.83 MB
    },
    "yolov8l.pt": {  # 大型模型（更高精度）
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l.pt",
        "size": 98406400  # 文件大小：~98.41 MB
    },
    "yolov8x.pt": {  # 超大模型（最高精度，适合GPU）
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt",
        "size": 163714560  # 文件大小：~163.71 MB
    }
}

# This file downloader demonstrates Streamlit animation.
def download_file(file_path):
    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(file_path):
        if "size" not in EXTERNAL_DEPENDENCIES[file_path]:
            return
        elif os.path.getsize(file_path) == EXTERNAL_DEPENDENCIES[file_path]["size"]:
            return
    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning(f"Downloading {file_path}... (Size: {EXTERNAL_DEPENDENCIES[file_path]['size']/1024/1024:.2f} MB)")
        progress_bar = st.progress(0)
        with open(file_path, "wb") as output_file:
            with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(f"Downloading {file_path}... ({counter / MEGABYTES:.2f}/{length / MEGABYTES:.2f} MB)")
                    progress_bar.progress(min(counter / length, 1.0))
    except Exception as e:
        st.error(f"Download failed: {str(e)}")
        print(e)
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

# This is the main app app itself, which appears when the user selects "Run the app".
def run_the_app():
    class Config():
        # 适配 YOLOv8：移除v4特有参数（tiny/phi），保留核心配置
        def __init__(self, weights='yolov8n.pt', shape=640, nms_iou=0.3, confidence=0.5):
            self.weights = weights
            self.shape = shape  # YOLOv8默认输入尺寸640
            self.confidence = confidence
            self.nms_iou = nms_iou

    # set title of app
    st.markdown('<h1 align="center">✌ Gesture Detection (YOLOv8)</h1>',
                unsafe_allow_html=True)
    st.sidebar.markdown("# Gesture Detection on?")
    activities = ["Example", "Image", "Camera", "FPS", "Heatmap", "Real Time", "Video"]
    choice = st.sidebar.selectbox("Choose among the given options:", activities)

    # -------------------------- 适配 YOLOv8 模型选择 --------------------------
    st.sidebar.markdown("# YOLOv8 Model Selection")
    # 提供v8模型选择（按尺寸从小到大）
    model_type = st.sidebar.selectbox(
        "Model Size (Speed ↓ Precision ↑)",
        [
            ("yolov8n.pt", "Nano (~6.26 MB, Fastest)"),
            ("yolov8s.pt", "Small (~24.56 MB, Balance)"),
            ("yolov8m.pt", "Medium (~56.83 MB, High Precision)"),
            ("yolov8l.pt", "Large (~98.41 MB, Higher Precision)"),
            ("yolov8x.pt", "X-Large (~163.71 MB, Highest Precision)")
        ],
        format_func=lambda x: x[1]
    )
    selected_weights = model_type[0]

    # YOLOv8输入尺寸选择（官方推荐640/1280）
    shape = st.sidebar.selectbox("Input Image Size", [640, 1280])
    conf, nms = object_detector_ui()

    @st.cache_resource  # 替换st.cache为st.cache_resource（适配模型缓存）
    def get_yolo(weights, conf, nms, shape=640):
        # YOLOv8加载方式（直接调用ultralytics.YOLO）
        yolo = YOLO(weights)
        # 设置模型参数
        yolo.model.conf = conf  # 置信度阈值
        yolo.model.iou = nms    # NMS IoU阈值
        return yolo

    # 加载YOLOv8模型
    yolo = get_yolo(selected_weights, conf, nms, shape)
    st.write(f"YOLOv8 Model Loaded: {selected_weights} (Input Size: {shape})")

    if choice == 'Image':
        detect_image(yolo)
    elif choice == 'Camera':
        detect_camera(yolo)
    elif choice == 'FPS':
        detect_fps(yolo, shape)
    elif choice == "Heatmap":
        detect_heatmap(yolo)
    elif choice == "Example":
        detect_example(yolo)
    elif choice == "Real Time":
        detect_realtime(yolo, shape)
    elif choice == "Video":
        detect_video(yolo, shape)

# This sidebar UI lets the user select parameters for the YOLO object detector.
def object_detector_ui():
    st.sidebar.markdown("# Model Parameters")
    confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    overlap_threshold = st.sidebar.slider("NMS Overlap threshold", 0.0, 1.0, 0.3, 0.01)
    return confidence_threshold, overlap_threshold

# -------------------------- 适配 YOLOv8 检测逻辑 --------------------------
def predict(image, yolo, shape=640):
    """Return predictions using YOLOv8."""
    try:
        # YOLOv8检测：指定输入尺寸，返回结果
        results = yolo.predict(image, imgsz=shape, conf=yolo.model.conf, iou=yolo.model.iou)
        # 绘制检测结果
        r_image = results[0].plot()  # 直接获取绘制后的图像（BGR格式）
        # 转换为RGB格式用于Streamlit显示
        r_image = cv2.cvtColor(r_image, cv2.COLOR_BGR2RGB)
        st.image(r_image, caption='Detected Image.', use_column_width=True)
    except Exception as e:
        st.error(f"Detection failed: {str(e)}")
        print(e)

def fps(image, yolo, shape=640):
    """Calculate FPS using YOLOv8."""
    test_interval = 10  # 减少测试次数，加快速度
    start_time = time.time()
    # 循环检测计算FPS
    for _ in range(test_interval):
        yolo.predict(image, imgsz=shape, conf=yolo.model.conf, iou=yolo.model.iou)
    end_time = time.time()
    tact_time = (end_time - start_time) / test_interval
    fps = 1 / tact_time
    st.write(f"Average inference time: {tact_time:.4f} seconds")
    st.write(f"FPS: {fps:.2f} (@batch_size 1)")
    return tact_time

# -------------------------- 以下函数仅适配YOLOv8调用逻辑，核心功能不变 --------------------------
def detect_image(yolo):
    file_up = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    classes = ["up", "down", "left", "right", "front", "back", "clockwise", "anticlockwise"]
    st.sidebar.markdown("See the model performance and play with it")
    if file_up is not None:
        with st.spinner(text='Preparing Image'):
            image = Image.open(file_up)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.balloons()
            detect = st.button("Start Detection")
            if detect:
                st.write("Processing...")
                predict(image, yolo)
                st.balloons()

def detect_camera(yolo):
    picture = st.camera_input("Take a picture")
    if picture:
        filters_to_funcs = {
            "No filter": predict,
            "Heatmap": heatmap,
            "FPS": fps,
        }
        filters = st.selectbox("Apply a filter!", filters_to_funcs.keys())
        image = Image.open(picture)
        with st.spinner(text='Preparing Image'):
            filters_to_funcs[filters](image, yolo)
            st.balloons()

def detect_fps(yolo, shape=640):
    file_up = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    st.sidebar.markdown("Test model FPS")
    if file_up is not None:
        image = Image.open(file_up)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.balloons()
        detect = st.button("Start FPS Test")
        if detect:
            with st.spinner(text='Calculating FPS...'):
                fps(image, yolo, shape)
                st.balloons()

def heatmap(image, yolo):
    """简化热力图功能（YOLOv8原生不支持，用检测结果替代）"""
    st.warning("YOLOv8 does not support heatmap natively, showing detection result instead")
    predict(image, yolo)

def detect_heatmap(yolo):
    file_up = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    st.sidebar.markdown("Generate Heatmap (Simulation)")
    if file_up is not None:
        image = Image.open(file_up)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.balloons()
        detect = st.button("Generate Heatmap")
        if detect:
            with st.spinner(text='Generating Heatmap...'):
                heatmap(image, yolo)
                st.balloons()

def detect_example(yolo):
    st.sidebar.title("Choose an Example Image")
    img_dir = './img'
    if os.path.exists(img_dir):
        images = os.listdir(img_dir)
        images = [img for img in images if img.endswith(('jpg', 'png', 'jpeg'))]
        if images:
            images.sort()
            selected_img = st.sidebar.selectbox("Image Name", images)
            image = Image.open(os.path.join(img_dir, selected_img))
            st.image(image, caption='Selected Example.', use_column_width=True)
            st.balloons()
            detect = st.button("Start Detection")
            if detect:
                st.write("Processing...")
                predict(image, yolo)
                st.balloons()
        else:
            st.warning("No example images found in ./img directory")
    else:
        st.warning("./img directory not found")

def detect_realtime(yolo, shape=640):
    class VideoProcessor:
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            # YOLOv8实时检测
            results = yolo.predict(img, imgsz=shape, conf=yolo.model.conf, iou=yolo.model.iou, verbose=False)
            r_image = results[0].plot()  # 绘制结果
            return av.VideoFrame.from_ndarray(r_image, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="gesture-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=False,
        video_processor_factory=VideoProcessor
    )

import cv2
import time
def detect_video(yolo, shape=640):
    file_up = st.file_uploader("Upload a video", type=["mp4"])
    if file_up is not None:
        video_path = 'temp_video.mp4'
        st.video(file_up)
        with open(video_path, 'wb') as f:
            f.write(file_up.read())
        detect = st.button("Start Video Detection")

        if detect:
            video_save_path = 'processed_video.mp4'
            capture = cv2.VideoCapture(video_path)
            video_fps = st.slider("Output Video FPS", 5, 30, int(capture.get(cv2.CAP_PROP_FPS)), 1)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 修正编码器，适配