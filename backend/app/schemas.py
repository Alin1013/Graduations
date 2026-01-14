"""
Pydantic模式定义（用于API请求/响应验证）
"""
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime


class VideoRecordBase(BaseModel):
    """视频记录基础模式"""
    filename: str
    file_path: str
    file_size: Optional[int] = None
    video_type: Optional[str] = None
    duration: Optional[float] = None
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None


class VideoRecordCreate(VideoRecordBase):
    """创建视频记录请求"""
    pass


class VideoRecordResponse(VideoRecordBase):
    """视频记录响应"""
    id: int
    upload_time: datetime
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class VideoUploadResponse(BaseModel):
    """视频上传响应"""
    video_id: int
    filename: str
    file_path: str
    video_type: Optional[str] = None
    upload_time: datetime
    message: str


class DetectionResultResponse(BaseModel):
    """检测结果响应"""
    id: int
    video_id: int
    output_video_path: str
    total_frames: Optional[int] = None
    detected_frames: Optional[int] = None
    total_detections: Optional[int] = None
    detection_summary: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None
    created_at: datetime
    
    class Config:
        from_attributes = True
