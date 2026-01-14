"""
视频服务模块
处理视频上传、保存等操作
"""
import os
import cv2
from datetime import datetime
from fastapi import UploadFile
from sqlalchemy.orm import Session
from app.models import VideoRecord
from app.database import get_db
from app.config import settings


async def save_uploaded_video(file: UploadFile) -> VideoRecord:
    """
    保存上传的视频文件并创建数据库记录
    """
    # 生成唯一文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_ext = os.path.splitext(file.filename)[1]
    filename = f"{timestamp}{file_ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    
    # 确保目录存在
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    # 保存文件
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    file_size = len(content)
    
    # 获取视频信息
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    
    # 创建数据库记录
    db = next(get_db())
    video_record = VideoRecord(
        filename=file.filename,
        file_path=file_path,
        file_size=file_size,
        video_type=file_ext[1:].lower() if file_ext else None,
        duration=duration,
        fps=fps,
        width=width,
        height=height,
        upload_time=datetime.now()
    )
    db.add(video_record)
    db.commit()
    db.refresh(video_record)
    
    return video_record
