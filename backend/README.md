# 手势检测后端服务

## 安装依赖

```bash
pip install -r requirements.txt
```

## 启动服务

```bash
python run.py
```

或者使用uvicorn直接启动：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## API文档

启动服务后，访问以下地址查看API文档：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 环境变量

可以通过环境变量配置以下参数：

- `DATABASE_URL`: 数据库连接URL（默认：SQLite）
- `UPLOAD_DIR`: 视频上传目录
- `OUTPUT_DIR`: 检测结果输出目录
- `DEFAULT_MODEL_PATH`: 默认模型路径
