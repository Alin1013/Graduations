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
from datetime import datetime

# 全局禁用SSL证书验证（解决模型下载HTTPS问题）
ssl._create_default_https_context = ssl._create_unverified_context

# 页面配置
st.set_page_config(
    page_title="手势检测平台",
    page_icon="✌️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# WebRTC配置（实时摄像头通信）
RTC_CONFIGURATION = RTCConfiguration({
    "RTCIceServers": [{
        "urls": ["stun:stun.l.google.com:19302"]
    }]
})

# YOLOv8模型配置（包含本地模型路径和自定义权重选项）
# 请在这里修改为你的本地模型文件路径
LOCAL_MODEL_PATHS = {
    "yolov8n.pt": "/Users/alin/Graduation_Project/yolov8n.pt",  # 替换为实际本地路径
    "yolov8s.pt": "/Users/alin/Graduation_Project/yolov8s.pt",  # 替换为实际本地路径
    "yolov8m.pt": "/Users/alin/Graduation_Project/yolov8m.pt",  # 替换为实际本地路径
    "yolov8l.pt": "/Users/alin/Graduation_Project/yolov8l.pt",  # 替换为实际本地路径
    "best.pt":"runs/detect/gesture_final_train/weights/best.pt",
}

MODEL_OPTIONS = {
    "yolov8n.pt": {"name": "目标识别-Nano (最快)", "local_path": LOCAL_MODEL_PATHS["yolov8n.pt"], "is_custom": False},
    "yolov8s.pt": {"name": "目标识别-Small (平衡)", "local_path": LOCAL_MODEL_PATHS["yolov8s.pt"], "is_custom": False},
    "yolov8m.pt": {"name": "目标识别-Medium (高精度)", "local_path": LOCAL_MODEL_PATHS["yolov8m.pt"], "is_custom": False},
    "yolov8l.pt": {"name": "目标识别-Large (超高精度)", "local_path": LOCAL_MODEL_PATHS["yolov8l.pt"], "is_custom": False},
    "custom_weight": {"name": "手势识别-Best (训练权重)", "local_path": LOCAL_MODEL_PATHS["best.pt"], "is_custom": False},  # 自定义权重占位符
}

# 支持的输入尺寸和手势类别
INPUT_SHAPES = [640, 1280]
GESTURE_CLASSES = ["one","two_up","two_up_inverted","three","four","fist","palm","ok","peace","loke","dislike","stop","stop_inverted","call","mute","rock","no_gesture"]

# 创建临时目录（存储上传的视频/权重）
os.makedirs("temp", exist_ok=True)

# -------------------------- 模型相关函数 --------------------------
def check_local_model(model_path):
    """检查本地模型文件是否存在"""
    if os.path.exists(model_path):
        st.info(f"已检测到本地模型！")
        return True
    else:
        st.error(f"本地模型文件不存在：{model_path}")
        return False

@st.cache_resource(show_spinner="加载模型中...")
def load_model(model_key, conf_threshold, nms_threshold, custom_weight_path=None):
    """加载YOLO模型（使用本地模型路径）"""
    model_info = MODEL_OPTIONS[model_key]

    # 加载官方模型（本地路径）
    if not model_info["is_custom"]:
        model_path = model_info["local_path"]
        if not check_local_model(model_path):
            return None
    # 加载自定义权重
    else:
        if not custom_weight_path or not os.path.exists(custom_weight_path):
            st.error("自定义权重文件不存在！")
            return None
        model_path = custom_weight_path

    try:
        model = YOLO(model_path)
        model.conf = conf_threshold  # 置信度阈值
        model.iou = nms_threshold    # NMS阈值
        return model
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return None

# -------------------------- 检测相关函数 --------------------------
def detect_image(model, image, input_shape):
    """单张图像检测"""
    if model is None:
        return None
    try:
        results = model.predict(image, imgsz=input_shape, verbose=False)
        return results[0].plot()  # 绘制检测框和类别
    except Exception as e:
        st.error(f"图像检测失败: {str(e)}")
        return None

class VideoProcessor:
    """实时摄像头视频处理类"""
    def __init__(self, model, input_shape):
        self.model = model
        self.input_shape = input_shape

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.model:
            results = self.model.predict(img, imgsz=self.input_shape, verbose=False)
            img = results[0].plot()
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def calculate_fps(model, input_shape):
    """计算模型推理FPS"""
    if model is None:
        return 0.0
    try:
        test_img = np.zeros((input_shape, input_shape, 3), dtype=np.uint8)
        start_time = time.time()
        # 多次推理取平均
        for _ in range(10):
            model.predict(test_img, imgsz=input_shape, verbose=False)
        elapsed = time.time() - start_time
        return 10 / elapsed
    except Exception as e:
        st.error(f"FPS计算失败: {str(e)}")
        return 0.0

def process_video(model, video_path, input_shape):
    """处理上传的视频并保存结果"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 输出视频路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"temp/gesture_detection_{timestamp}.mp4"

    # 视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # 处理进度条
    progress_bar = st.progress(0)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 检测并绘制结果
        results = model.predict(frame, imgsz=input_shape, verbose=False)
        result_frame = results[0].plot()
        out.write(result_frame)

        # 更新进度
        frame_idx += 1
        progress_bar.progress(min(frame_idx / frame_count, 1.0))

    # 释放资源
    cap.release()
    out.release()
    progress_bar.empty()
    return output_path

# -------------------------- 主应用函数 --------------------------
def main():
    st.title("✌️ 手势检测平台")
    st.markdown("基于YOLOv8的实时手势检测系统 | 支持图像/摄像头/视频三种检测模式")

    # 侧边栏配置
    with st.sidebar:
        st.header("🔧 参数设置")

        # 1. 应用模式选择
        app_mode = st.selectbox(
            "选择功能模式",
            ["图像检测", "实时摄像头", "视频上传", "性能测试", "关于"]
        )

        # 2. 模型设置
        st.subheader("📦 模型配置")
        model_key = st.selectbox(
            "选择模型",
            list(MODEL_OPTIONS.keys()),
            format_func=lambda x: MODEL_OPTIONS[x]["name"]
        )

        # 自定义权重上传（仅当选择自定义权重时显示）
        custom_weight_path = None
        if MODEL_OPTIONS[model_key]["is_custom"]:
            st.warning("请上传训练好的手势检测权重文件（.pt格式）")
            uploaded_weight = st.file_uploader("上传自定义权重", type=["pt"])

            if uploaded_weight is not None:
                # 保存上传的权重到临时文件
                custom_weight_path = "temp/custom_best.pt"
                with open(custom_weight_path, "wb") as f:
                    f.write(uploaded_weight.getbuffer())
                st.success("自定义权重上传成功！")
            else:
                # 检查默认路径是否有权重（兼容原有训练路径）
                default_custom_path = "runs/detect/normal_train/weights/best.pt"
                if os.path.exists(default_custom_path):
                    custom_weight_path = default_custom_path
                    st.info(f"找到默认路径权重：{default_custom_path}")

        # 3. 检测参数
        input_shape = st.selectbox(
            "输入图像尺寸",
            INPUT_SHAPES,
            format_func=lambda x: f"{x}x{x}"
        )

        conf_threshold = st.slider(
            "置信度阈值",
            0.0, 1.0, 0.5, 0.01,
            help="检测框置信度过滤（值越高越严格）"
        )

        nms_threshold = st.slider(
            "NMS阈值",
            0.0, 1.0, 0.3, 0.01,
            help="去除重叠检测框的阈值"
        )

        # 4. 加载模型按钮
        st.markdown("---")
        load_btn = st.button("🚀 加载模型")
        if load_btn:
            with st.spinner("正在加载模型..."):
                # 加载模型（根据选择的模型类型传入不同参数）
                if MODEL_OPTIONS[model_key]["is_custom"]:
                    model = load_model(
                        model_key,
                        conf_threshold,
                        nms_threshold,
                        custom_weight_path=custom_weight_path
                    )
                else:
                    model = load_model(
                        model_key,
                        conf_threshold,
                        nms_threshold
                    )

                if model:
                    st.session_state["model"] = model
                    st.session_state["model_info"] = MODEL_OPTIONS[model_key]
                    st.success(f"✅ 模型加载成功：{MODEL_OPTIONS[model_key]['name']}")
                else:
                    st.session_state["model"] = None
                    st.error("❌ 模型加载失败，请检查配置！")

        # 5. 支持的手势
        st.markdown("---")
        st.subheader("🖐️ 支持手势")
        st.write(" | ".join(GESTURE_CLASSES))

    # 检查模型是否加载（除"关于"模式外）
    model = st.session_state.get("model")
    if model is None and app_mode not in ["关于"]:
        st.warning("⚠️ 请先在侧边栏选择模型并点击'加载模型'按钮")
        return

    # -------------------------- 功能模式实现 --------------------------
    if app_mode == "图像检测":
        st.subheader("📷 图像检测")
        st.write("上传包含手势的图像，系统将自动识别并标记类别")

        uploaded_file = st.file_uploader("选择图像文件", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # 显示原始图像
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)

            with col1:
                st.info("原始图像")
                st.image(image, use_column_width=True)

            # 检测按钮
            if st.button("开始检测"):
                with st.spinner("正在处理图像..."):
                    img_array = np.array(image)
                    result_img = detect_image(model, img_array, input_shape)

                    with col2:
                        st.success("检测结果")
                        if result_img is not None:
                            st.image(result_img, use_column_width=True)

    elif app_mode == "实时摄像头":
        st.subheader("📹 实时摄像头检测")
        st.info("点击下方区域启动摄像头（首次使用需授予浏览器权限）")

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
        st.subheader("🎥 视频上传检测")
        st.warning("提示：视频处理时间取决于视频长度和设备性能，建议先测试短视频（<1分钟）")

        uploaded_video = st.file_uploader("选择视频文件", type=["mp4", "mov", "avi"])
        if uploaded_video is not None:
            # 保存上传的视频到临时文件
            temp_video_path = "temp/uploaded_video.mp4"
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_video.read())

            # 显示视频信息
            cap = cv2.VideoCapture(temp_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            st.info(f"视频信息：{fps:.1f} FPS | {frame_count} 帧 | 时长：{duration:.1f} 秒")
            cap.release()

            # 显示原始视频预览
            st.subheader("原始视频预览")
            st.video(temp_video_path)

            # 处理视频按钮
            if st.button("开始处理视频"):
                with st.spinner("正在处理视频..."):
                    output_path = process_video(model, temp_video_path, input_shape)
                    st.success("✅ 视频处理完成！")

                    # 显示处理结果
                    st.subheader("处理结果预览")
                    st.video(output_path)

                    # 下载按钮
                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="下载处理后的视频",
                            data=f,
                            file_name=f"gesture_detection_{datetime.now().strftime('%Y%m%d')}.mp4",
                            mime="video/mp4"
                        )

    elif app_mode == "性能测试":
        st.subheader("⚡ 模型性能测试")
        st.write("测试当前模型在设备上的推理速度（FPS），结果仅供参考")

        if st.button("开始测试FPS"):
            with st.spinner("正在测试性能..."):
                fps = calculate_fps(model, input_shape)
                st.success(f"测试完成！平均FPS：{fps:.2f} 帧/秒")

                # 性能评估
                if fps < 10:
                    st.warning("性能较低，建议优化：\n1. 选择更小的模型（如yolov8n）\n2. 降低输入尺寸（如640x640）\n3. 使用GPU加速")
                elif fps < 25:
                    st.info("性能中等，可满足基本实时检测需求")
                else:
                    st.success("性能优异，适合高质量实时检测！")

    elif app_mode == "关于":
        st.subheader("📋 关于本平台")
        st.markdown("""
        ### 手势检测平台（基于YOLOv8）
        
        **核心功能**：
        - 图像检测：单张图像手势识别
        - 实时摄像头：浏览器端实时手势跟踪
        - 视频上传：批量处理视频并保存检测结果
        - 性能测试：评估模型在当前设备的运行速度
        
        **技术栈**：
        - 目标检测：YOLOv8（Ultralytics）
        - Web框架：Streamlit
        - 实时通信：WebRTC
        - 图像处理：OpenCV、NumPy
        
        **使用提示**：
        1. 建议在光线充足的环境下使用，提高检测准确率
        2. 手势尽量清晰可见，避免复杂背景干扰
        3. 自定义权重需使用YOLOv8训练的手势检测模型（8类手势）
        4. 实时检测建议FPS≥15，可通过调整模型和输入尺寸优化
        """)

if __name__ == "__main__":
    main()