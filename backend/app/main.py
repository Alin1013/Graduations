"""
手势检测后端API主文件
基于FastAPI框架
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from datetime import datetime
from typing import List, Optional

from contextlib import asynccontextmanager
from app.database import init_db, get_db
from app.models import VideoRecord, DetectionResult
from app.schemas import (
    VideoRecordCreate, VideoRecordResponse,
    DetectionResultResponse, VideoUploadResponse
)
from app.services import video_service, detection_service
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    init_db()
    print("数据库初始化完成")
    yield
    # 关闭时（可以在这里添加清理代码）


# 创建FastAPI应用
app = FastAPI(
    title="手势检测API",
    description="基于YOLOv8的手势检测后端服务",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应设置为具体的前端地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 确保必要的目录存在
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

# 挂载静态文件目录（用于提供视频文件访问）
app.mount("/static", StaticFiles(directory=settings.OUTPUT_DIR), name="static")
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")


@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "手势检测API服务",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/api/videos/upload",
            "detect": "/api/videos/{video_id}/detect",
            "records": "/api/videos/records",
            "result": "/api/detections/{detection_id}"
        }
    }


@app.post("/api/videos/upload", response_model=VideoUploadResponse)
async def upload_video(file: UploadFile = File(...)):
    """
    上传视频文件
    """
    try:
        # 验证文件类型
        if not file.filename.endswith(('.mp4', '.mov', '.avi', '.mkv')):
            raise HTTPException(status_code=400, detail="不支持的视频格式，请上传mp4/mov/avi/mkv文件")
        
        # 保存上传的视频
        video_record = await video_service.save_uploaded_video(file)
        
        return VideoUploadResponse(
            video_id=video_record.id,
            filename=video_record.filename,
            file_path=video_record.file_path,
            video_type=video_record.video_type,
            upload_time=video_record.upload_time,
            message="视频上传成功"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"视频上传失败: {str(e)}")


@app.post("/api/videos/{video_id}/detect")
async def detect_video(
    video_id: int,
    model_path: Optional[str] = None,
    conf_threshold: float = 0.5,
    nms_threshold: float = 0.3,
    input_shape: int = 640
):
    """
    对上传的视频进行手势检测
    """
    try:
        # 获取视频记录
        db = next(get_db())
        video_record = db.query(VideoRecord).filter(VideoRecord.id == video_id).first()
        if not video_record:
            raise HTTPException(status_code=404, detail="视频记录不存在")
        
        # 执行检测
        detection_result = await detection_service.detect_video(
            video_record=video_record,
            model_path=model_path or settings.DEFAULT_MODEL_PATH,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            input_shape=input_shape
        )
        
        return DetectionResultResponse.model_validate(detection_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"视频检测失败: {str(e)}")


@app.get("/api/videos/records", response_model=List[VideoRecordResponse])
async def get_video_records(skip: int = 0, limit: int = 100):
    """
    获取所有视频记录列表
    """
    try:
        db = next(get_db())
        records = db.query(VideoRecord).offset(skip).limit(limit).all()
        return [VideoRecordResponse.model_validate(record) for record in records]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取视频记录失败: {str(e)}")


@app.get("/api/videos/{video_id}", response_model=VideoRecordResponse)
async def get_video_record(video_id: int):
    """
    获取单个视频记录的详细信息
    """
    try:
        db = next(get_db())
        record = db.query(VideoRecord).filter(VideoRecord.id == video_id).first()
        if not record:
            raise HTTPException(status_code=404, detail="视频记录不存在")
        return VideoRecordResponse.model_validate(record)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取视频记录失败: {str(e)}")


@app.get("/api/detections/{detection_id}", response_model=DetectionResultResponse)
async def get_detection_result(detection_id: int):
    """
    获取检测结果详情
    """
    try:
        db = next(get_db())
        result = db.query(DetectionResult).filter(DetectionResult.id == detection_id).first()
        if not result:
            raise HTTPException(status_code=404, detail="检测结果不存在")
        return DetectionResultResponse.model_validate(result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取检测结果失败: {str(e)}")


@app.get("/api/videos/{video_id}/detections", response_model=List[DetectionResultResponse])
async def get_video_detections(video_id: int):
    """
    获取某个视频的所有检测结果
    """
    try:
        db = next(get_db())
        results = db.query(DetectionResult).filter(
            DetectionResult.video_id == video_id
        ).order_by(DetectionResult.created_at.desc()).all()
        return [DetectionResultResponse.model_validate(result) for result in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取检测结果失败: {str(e)}")


@app.get("/api/videos/{video_id}/download")
async def download_processed_video(video_id: int, detection_id: Optional[int] = None):
    """
    下载处理后的视频文件
    """
    try:
        db = next(get_db())
        
        if detection_id:
            # 下载指定检测结果的视频
            result = db.query(DetectionResult).filter(DetectionResult.id == detection_id).first()
            if not result:
                raise HTTPException(status_code=404, detail="检测结果不存在")
            video_path = result.output_video_path
        else:
            # 下载最新的检测结果视频
            result = db.query(DetectionResult).filter(
                DetectionResult.video_id == video_id
            ).order_by(DetectionResult.created_at.desc()).first()
            if not result:
                raise HTTPException(status_code=404, detail="该视频暂无检测结果")
            video_path = result.output_video_path
        
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail="视频文件不存在")
        
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename=os.path.basename(video_path)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"下载视频失败: {str(e)}")


@app.delete("/api/videos/{video_id}")
async def delete_video_record(video_id: int):
    """
    删除视频记录及其相关检测结果
    """
    try:
        db = next(get_db())
        record = db.query(VideoRecord).filter(VideoRecord.id == video_id).first()
        if not record:
            raise HTTPException(status_code=404, detail="视频记录不存在")
        
        # 删除相关的检测结果
        detections = db.query(DetectionResult).filter(DetectionResult.video_id == video_id).all()
        for detection in detections:
            # 删除输出视频文件
            if os.path.exists(detection.output_video_path):
                os.remove(detection.output_video_path)
            db.delete(detection)
        
        # 删除原始视频文件
        if os.path.exists(record.file_path):
            os.remove(record.file_path)
        
        # 删除记录
        db.delete(record)
        db.commit()
        
        return {"message": "视频记录及相关文件已删除"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
