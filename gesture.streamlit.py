"""手势检测web平台"""
import os
import time
import cv2
import numpy as np
import torch
from PIL import Image
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
from ultralytics import YOLO
import ssl
# 全局禁用SSL证书验证
ssl._create_default_https_context = ssl._create_unverified_context
# 页面配置
st.set_page_config(
    page_title="手势检测平台",
    page_icon="✌️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# WebRTC配置
RTC_CONFIGURATION = RTCConfiguration({
    "RTCIceServers": [{
        "urls": ["stun:stun.l.google.com:19302"]
    }]
})

# YOLOv8模型配置
MODEL_OPTIONS = {
    "yolov8n.pt": {"name": "Nano (最快)", "size": 6257408},
    "yolov8s.pt": {"name": "Small (平衡)", "size": 24555520},
    "yolov8m.pt": {"name": "Medium (高精度)", "size": 56834560},
    "yolov8l.pt": {"name": "Large (超高精度)", "size": 98406400},
    "yolov8x.pt": {"name": "X-Large (最高精度)", "size": 163714560}
}

INPUT_SHAPES = [640, 1280]
GESTURE_CLASSES = ["up", "down", "left", "right", "front", "back", "clockwise", "anticlockwise"]

# 下载模型权重 - 仅使用一个进度条
def download_model(model_path):
    if os.path.exists(model_path) and os.path.getsize(model_path) == MODEL_OPTIONS[model_path]["size"]:
        return True

    try:
        # 创建进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text(f"开始下载 {MODEL_OPTIONS[model_path]['name']} 模型...")

        model_url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{model_path}"

        # 自定义下载进度回调函数
        def update_progress(count, block_size, total_size):
            progress = min(count * block_size / total_size, 1.0)
            progress_bar.progress(progress)
            # 不显示额外的文字信息，只保留进度条

        # 使用urllib下载
        import urllib.request
        urllib.request.urlretrieve(
            model_url,
            model_path,
            reporthook=update_progress
        )

        # 下载完成后清理进度条
        progress_bar.empty()
        status_text.empty()
        return os.path.exists(model_path)
    except Exception as e:
        st.error(f"模型下载失败: {str(e)}")
        return False

# 加载YOLO模型
@st.cache_resource
def load_model(model_path, conf_threshold, nms_threshold):
    if not download_model(model_path):
        return None

    try:
        model = YOLO(model_path)
        model.conf = conf_threshold  # 置信度阈值
        model.iou = nms_threshold    # NMS阈值
        return model
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return None

# 图像检测函数
def detect_image(model, image, input_shape):
    if model is None:
        return None

    try:
        results = model.predict(image, imgsz=input_shape)
        return results[0].plot()  # 返回绘制了检测结果的图像
    except Exception as e:
        st.error(f"检测失败: {str(e)}")
        return None

# 视频处理类
class VideoProcessor:
    def __init__(self, model, input_shape):
        self.model = model
        self.input_shape = input_shape
        self.conf_threshold = model.conf if model else 0.5

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if self.model:
            results = self.model.predict(img, imgsz=self.input_shape)
            img = results[0].plot()

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 计算FPS
def calculate_fps(model, input_shape):
    if model is None:
        return 0.0

    try:
        test_img = np.zeros((input_shape, input_shape, 3), dtype=np.uint8)
        start_time = time.time()
        for _ in range(10):
            model.predict(test_img, imgsz=input_shape)
        elapsed = time.time() - start_time
        return 10 / elapsed
    except Exception as e:
        st.error(f"FPS计算失败: {str(e)}")
        return 0.0

# 主应用函数
def main():
    # 页面标题
    st.title("✌️ 手势检测平台")
    st.markdown("实时检测手势动作，支持图像上传、摄像头实时检测等多种模式")

    # 侧边栏配置
    with st.sidebar:
        st.header("设置")

        # 选择应用模式
        app_mode = st.selectbox(
            "应用模式",
            ["图像检测", "实时摄像头", "视频上传", "性能测试", "关于"]
        )

        # 模型设置
        st.subheader("模型设置")
        model_name = st.selectbox(
            "选择模型",
            list(MODEL_OPTIONS.keys()),
            format_func=lambda x: f"{MODEL_OPTIONS[x]['name']} ({x})"
        )

        input_shape = st.selectbox(
            "输入图像尺寸",
            INPUT_SHAPES,
            format_func=lambda x: f"{x}x{x}"
        )

        conf_threshold = st.slider(
            "置信度阈值",
            0.0, 1.0, 0.5, 0.01
        )

        nms_threshold = st.slider(
            "NMS阈值",
            0.0, 1.0, 0.3, 0.01
        )

        # 加载模型按钮
        if st.button("加载模型"):
            with st.spinner("正在加载模型..."):
                st.session_state["model"] = load_model(
                    model_name,
                    conf_threshold,
                    nms_threshold
                )
                if st.session_state.get("model"):
                    st.success(f"模型 {MODEL_OPTIONS[model_name]['name']} 加载成功!")

        # 展示手势识别
        st.subheader("支持的手势")
        st.write(", ".join(GESTURE_CLASSES))

    # 确保模型已加载
    model = st.session_state.get("model")
    if model is None and app_mode not in ["关于"]:
        st.warning("请先在侧边栏选择模型并点击'加载模型'按钮")
        return

    # 根据选择的模式展示对应内容
    if app_mode == "图像检测":
        st.subheader("图像检测")
        uploaded_file = st.file_uploader("上传图像", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # 展示原始图像
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)

            with col1:
                st.info("原始图像")
                st.image(image, use_column_width=True)

            # 检测图像
            if st.button("开始检测"):
                with st.spinner("正在檢測..."):
                    # 转换图像格式
                    img_array = np.array(image)
                    result_img = detect_image(model, img_array, input_shape)

                    with col2:
                        st.success("检测结果")
                        if result_img is not None:
                            st.image(result_img, use_column_width=True)

    elif app_mode == "实时摄像头":
        st.subheader("实时摄像头检测")
        st.info("点击下方按钮启动摄像头，请确保浏览器已授予摄像头权限")

        if model:
            webrtc_streamer(
                key="gesture-detection",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=lambda: VideoProcessor(model, input_shape),
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

    elif app_mode == "视频上传":
        st.subheader("视频上传检测")
        st.warning("注意：视频检测可能需要较长时间，取决于视频长度和模型性能")

        uploaded_video = st.file_uploader("上传视频", type=["mp4", "mov", "avi"])

        if uploaded_video is not None:
            # 保存上传的视频
            video_path = "temp_video.mp4"
            with open(video_path, "wb") as f:
                f.write(uploaded_video.read())

            # 展示视频信息
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0

            st.info(f"视频信息: {fps:.1f} FPS, {frame_count} 帧, 约 {duration:.1f} 秒")
            cap.release()

            if st.button("开始处理视频"):
                with st.spinner("正在处理视频..."):
                    # 处理视频
                    cap = cv2.VideoCapture(video_path)
                    output_frames = []

                    # 展示处理进度
                    progress_bar = st.progress(0)
                    frame_idx = 0

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # 检测每一帧
                        result_frame = detect_image(model, frame, input_shape)
                        if result_frame is not None:
                            output_frames.append(result_frame)

                        # 更新进度
                        frame_idx += 1
                        progress_bar.progress(min(frame_idx / frame_count, 1.0))

                    cap.release()
                    progress_bar.empty()

                    st.success("视频处理完成!")
                    # 这里可以添加视频保存和演示代码

    elif app_mode == "性能测试":
        st.subheader("性能测试")
        st.info("测试模型在当前设备上的推理速度")

        if st.button("开始测试FPS"):
            with st.spinner("正在测试..."):
                fps = calculate_fps(model, input_shape)
                st.success(f"平均FPS: {fps:.2f} 帧/秒")

                # 展示性能评估
                if fps < 10:
                    st.warning("性能较低，建议使用更小的模型或降低输入尺寸")
                elif fps < 25:
                    st.info("性能中等，可以满足基本实时需求")
                else:
                    st.success("性能优异，适合实时检测")

    elif app_mode == "关于":
        st.subheader("关于本应用")
        st.markdown("""
        这是一个基于YOLOv8的手势检测Web应用，可以识别小物体目标检测，同时支持多种手势的实时检测。
        
        ### 支持的手势
        - 上下左右 (up, down, left, right)
        - 前后 (front, back)
        - 顺时针/逆时针旋转 (clockwise, anticlockwise)
        
        ### 使用说明
        1. 在侧边栏选择合适的模型和参数
        2. 点击“加载模型”按钮加载模型
        3. 选择相应的应用模式进行检测
        
        ### 注意事项
        - 较大的模型精度更高但速度更慢
        - 较大的输入尺寸可能提高精度但降低速度
        - 实时检测需要较高的FPS (建议至少15 FPS)
        """)

if __name__ == "__main__":
    main()