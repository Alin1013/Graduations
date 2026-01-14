"""
数据库模型定义
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
import json
from app.database import Base


class VideoRecord(Base):
    """视频记录表"""
    __tablename__ = "video_records"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False, comment="原始文件名")
    file_path = Column(String(512), nullable=False, comment="视频文件存储路径")
    file_size = Column(Integer, comment="文件大小（字节）")
    video_type = Column(String(50), comment="视频类型（mp4/mov/avi等）")
    duration = Column(Float, comment="视频时长（秒）")
    fps = Column(Float, comment="视频帧率")
    width = Column(Integer, comment="视频宽度")
    height = Column(Integer, comment="视频高度")
    upload_time = Column(DateTime, default=datetime.now, comment="上传时间")
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment="更新时间")
    
    # 关联关系
    detections = relationship("DetectionResult", back_populates="video", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<VideoRecord(id={self.id}, filename='{self.filename}')>"


class DetectionResult(Base):
    """检测结果表"""
    __tablename__ = "detection_results"
    
    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("video_records.id"), nullable=False, comment="关联的视频ID")
    output_video_path = Column(String(512), nullable=False, comment="处理后的视频路径")
    
    # 检测统计信息
    total_frames = Column(Integer, comment="总帧数")
    detected_frames = Column(Integer, comment="检测到目标的帧数")
    total_detections = Column(Integer, comment="总检测数")
    detection_summary = Column(JSON, comment="检测结果摘要（各类别统计）")
    
    # 处理信息
    model_path = Column(String(512), comment="使用的模型路径")
    conf_threshold = Column(Float, comment="置信度阈值")
    nms_threshold = Column(Float, comment="NMS阈值")
    input_shape = Column(Integer, comment="输入图像尺寸")
    processing_time = Column(Float, comment="处理耗时（秒）")
    
    # 时间戳
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")
    
    # 关联关系
    video = relationship("VideoRecord", back_populates="detections")
    
    def __repr__(self):
        return f"<DetectionResult(id={self.id}, video_id={self.video_id}, total_detections={self.total_detections})>"
