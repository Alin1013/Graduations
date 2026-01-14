"""
检测服务模块
处理视频检测、结果保存等操作
"""
import os
import cv2
import time
import json
from datetime import datetime
from typing import Dict, Any
from ultralytics import YOLO
from sqlalchemy.orm import Session
from app.models import VideoRecord, DetectionResult
from app.database import get_db
from app.config import settings


async def detect_video(
    video_record: VideoRecord,
    model_path: str,
    conf_threshold: float = 0.5,
    nms_threshold: float = 0.3,
    input_shape: int = 640
) -> DetectionResult:
    """
    对视频进行手势检测并保存结果
    """
    start_time = time.time()
    
    # 加载模型
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    model = YOLO(model_path)
    model.conf = conf_threshold
    model.iou = nms_threshold
    
    # 打开视频
    cap = cv2.VideoCapture(video_record.file_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_record.file_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps <= 0:
        fps = 25.0
    
    # 创建输出视频路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    output_filename = f"detection_{video_record.id}_{timestamp}.mp4"
    output_path = os.path.join(settings.OUTPUT_DIR, output_filename)
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    if not out.isOpened():
        # 尝试其他编码格式
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        if not out.isOpened():
            raise RuntimeError(f"无法创建输出视频文件: {output_path}")
    
    # 统计信息
    detected_frames = 0
    total_detections = 0
    class_counts: Dict[str, int] = {}
    frame_detections = []
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 执行检测
        results = model.predict(frame, imgsz=input_shape, verbose=False)
        result_frame = results[0].plot()
        out.write(result_frame)
        
        # 统计检测结果
        frame_detections_list = []
        if results[0].boxes is not None:
            detected_frames += 1
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model.names[cls_id] if hasattr(model, 'names') else f"class_{cls_id}"
                
                total_detections += 1
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                
                frame_detections_list.append({
                    "frame": frame_idx,
                    "class_name": cls_name,
                    "confidence": conf
                })
        
        frame_detections.append({
            "frame": frame_idx,
            "detections": frame_detections_list
        })
        
        frame_idx += 1
    
    # 释放资源
    cap.release()
    out.release()
    
    processing_time = time.time() - start_time
    
    # 创建检测结果摘要
    detection_summary = {
        "class_counts": class_counts,
        "frame_detections": frame_detections[:100]  # 只保存前100帧的详细数据
    }
    
    # 保存到数据库
    db = next(get_db())
    detection_result = DetectionResult(
        video_id=video_record.id,
        output_video_path=output_path,
        total_frames=frame_count,
        detected_frames=detected_frames,
        total_detections=total_detections,
        detection_summary=detection_summary,
        model_path=model_path,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
        input_shape=input_shape,
        processing_time=processing_time
    )
    db.add(detection_result)
    db.commit()
    db.refresh(detection_result)
    
    return detection_result
