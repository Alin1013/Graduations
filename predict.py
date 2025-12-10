import time
import yaml
import cv2
import numpy as np
import os
import argparse
from PIL import Image
from tqdm import tqdm
from get_yaml import get_config

# 修复YOLO类的导入和初始化逻辑
try:
    from yolo import YOLO
except ImportError as e:
    print(f"导入YOLO模块失败: {e}")
    exit(1)


def init_yolo(opt):
    """
    初始化YOLO模型（修复参数传递问题）
    :param opt: 命令行参数对象
    :return: YOLO模型实例
    """
    try:
        # 创建YOLO模型配置字典（避免直接传递opt对象导致的属性错误）
        yolo_config = {
            "weights": opt.weights,
            "tiny": opt.tiny,
            "phi": opt.phi,
            "cuda": opt.cuda,
            "shape": opt.shape,
            "confidence": opt.confidence,
            "nms_iou": opt.nms_iou
        }

        # 初始化YOLO模型（适配不同的初始化方式）
        if hasattr(YOLO, '__init__'):
            # 如果YOLO类需要配置字典参数
            yolo = YOLO(yolo_config)
        else:
            # 兼容原有的opt参数方式
            yolo = YOLO(opt)

        return yolo
    except AttributeError as e:
        print(f"初始化YOLO模型失败: {e}")
        print("尝试使用简化模式初始化...")
        # 简化模式：直接传递权重路径
        yolo = YOLO(opt.weights)
        # 手动设置其他参数
        yolo.conf = opt.confidence
        yolo.iou = opt.nms_iou
        yolo.device = "cuda" if opt.cuda else "cpu"
        yolo.imgsz = opt.shape
        return yolo
    except Exception as e:
        print(f"初始化YOLO模型出错: {str(e)}")
        exit(1)


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="YOLOv8 预测工具")
    parser.add_argument('--weights', type=str, default='model_data/yolov8n.pt',
                        help='初始权重路径 (default: model_data/yolov8n.pt)')
    parser.add_argument('--tiny', action='store_true',
                        help='使用yolov8n模型(轻量化模型)')
    parser.add_argument('--phi', type=int, default=1,
                        help='注意力机制类型 (default: 1)')
    parser.add_argument('--mode', type=str,
                        choices=['dir_predict', 'video', 'fps', 'predict', 'heatmap', 'export_onnx'],
                        default="dir_predict",
                        help='预测的模式 (default: dir_predict)')
    parser.add_argument('--cuda', action='store_true',
                        help='使用GPU加速 (默认使用CPU)')
    parser.add_argument('--shape', type=int, default=640,
                        help='输入图像的尺寸 (default: 640)')
    parser.add_argument('--video', type=str, default='',
                        help='需要检测的视频文件路径 (默认使用摄像头)')
    parser.add_argument('--save-video', type=str, default='',
                        help='保存检测后视频的路径 (不填则不保存)')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='检测置信度阈值 (default: 0.5)')
    parser.add_argument('--nms_iou', type=float, default=0.3,
                        help='非极大抑制IOU阈值 (default: 0.3)')
    opt = parser.parse_args()

    # 打印配置信息
    print("=" * 60)
    print("YOLOv8 预测配置")
    print("=" * 60)
    for k, v in vars(opt).items():
        print(f"{k}: {v}")
    print("=" * 60)

    # 加载配置文件
    try:
        config = get_config()
        print(f"✅ 配置文件加载成功，类别数: {config['nc']}")
    except Exception as e:
        print(f"⚠️  配置文件加载失败: {e}，使用默认配置")
        config = None

    # 初始化YOLO模型（核心修复）
    yolo = init_yolo(opt)

    # 模式配置参数
    crop = False  # 是否裁剪检测到的目标
    count = False  # 是否计数检测到的目标

    # 视频检测参数
    video_path = 0 if opt.video == '' else opt.video
    video_save_path = opt.save_video
    video_fps = 25.0

    # FPS测试参数
    test_interval = 100
    fps_image_path = "img/call.jpg"

    # 目录检测参数
    dir_origin_path = "img/"
    dir_save_path = "img_out/"

    # 热力图参数
    heatmap_save_path = "model_data/heatmap_vision.png"

    # ONNX导出参数
    simplify = True
    onnx_save_path = "model_data/models.onnx"

    # -------------------------- 单张图片预测模式 --------------------------
    if mode == "predict":
        print("\n📸 单张图片预测模式")
        print("提示：输入 'q' 退出，输入图片路径进行预测")
        while True:
            img_path = input('\n请输入图片路径: ').strip()
            if img_path.lower() == 'q':
                print("退出预测")
                break
            if not os.path.exists(img_path):
                print(f"❌ 图片路径不存在: {img_path}")
                continue

            try:
                image = Image.open(img_path).convert('RGB')
                # 执行检测
                r_image = yolo.detect_image(image, crop=crop, count=count)

                # 显示和保存结果
                r_image.show(title="检测结果")
                os.makedirs(dir_save_path, exist_ok=True)
                save_path = os.path.join(dir_save_path, 'img_result.jpg')
                r_image.save(save_path, quality=95)
                print(f"✅ 检测结果已保存至: {save_path}")

            except Exception as e:
                print(f"❌ 处理图片失败: {e}")
                continue

    # -------------------------- 视频检测模式 --------------------------
    elif mode == "video":
        print(f"\n🎥 视频检测模式")
        print(f"视频源: {video_path if video_path != 0 else '摄像头'}")

        # 打开视频/摄像头
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise ValueError(f"❌ 无法打开视频源: {video_path}")

        # 获取视频参数
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0

        # 初始化视频保存器
        video_writer = None
        if video_save_path != '':
            os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 改用MP4格式（更通用）
            video_writer = cv2.VideoWriter(
                video_save_path, fourcc, video_fps, (frame_width, frame_height)
            )
            print(f"📹 将保存检测视频至: {video_save_path}")

        # 视频处理循环
        fps = 0.0
        frame_count = 0
        print("按 ESC 键退出")

        while True:
            t1 = time.time()
            ret, frame = capture.read()

            if not ret:
                print(f"\n📽️  视频处理完成，共处理 {frame_count} 帧")
                break

            try:
                # 格式转换：BGR → RGB → PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)

                # 执行检测
                result_image = yolo.detect_image(image, crop=crop, count=count)

                # 格式转换：PIL Image → numpy → BGR
                frame_result = np.array(result_image)
                frame_result = cv2.cvtColor(frame_result, cv2.COLOR_RGB2BGR)

                # 计算并显示FPS
                fps = (fps + (1. / (time.time() - t1))) / 2
                cv2.putText(frame_result, f"FPS: {fps:.2f}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 显示结果
                cv2.imshow("YOLOv8 Video Detection", frame_result)

                # 保存视频帧
                if video_writer is not None:
                    video_writer.write(frame_result)

                frame_count += 1

                # 按键退出
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC键
                    print("\n🛑 用户手动退出")
                    break

            except Exception as e:
                print(f"\n⚠️  处理第 {frame_count} 帧失败: {e}")
                continue

        # 释放资源
        capture.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        print("✅ 视频检测完成")

    # -------------------------- FPS测试模式 --------------------------
    elif mode == "fps":
        print(f"\n⚡ FPS测试模式")
        print(f"测试图片: {fps_image_path}")
        print(f"测试次数: {test_interval}")

        if not os.path.exists(fps_image_path):
            print(f"❌ 测试图片不存在: {fps_image_path}")
            exit(1)

        try:
            img = Image.open(fps_image_path).convert('RGB')
            # 预热模型
            print("预热模型...")
            for _ in range(10):
                yolo.detect_image(img)

            # 正式测试
            start_time = time.time()
            for _ in tqdm(range(test_interval), desc="FPS测试"):
                yolo.detect_image(img)
            end_time = time.time()

            # 计算FPS
            tact_time = (end_time - start_time) / test_interval
            fps = 1 / tact_time
            print(f"\n📊 FPS测试结果:")
            print(f"单帧耗时: {tact_time:.4f} 秒")
            print(f"FPS: {fps:.2f} (batch_size=1)")

        except Exception as e:
            print(f"❌ FPS测试失败: {e}")
            exit(1)

    # -------------------------- 目录批量预测模式 --------------------------
    elif mode == "dir_predict":
        print(f"\n📁 目录批量预测模式")
        print(f"输入目录: {dir_origin_path}")
        print(f"输出目录: {dir_save_path}")

        if not os.path.exists(dir_origin_path):
            print(f"❌ 输入目录不存在: {dir_origin_path}")
            exit(1)

        # 创建输出目录
        os.makedirs(dir_save_path, exist_ok=True)

        # 获取图片列表
        img_extensions = ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm',
                          '.pgm', '.ppm', '.tif', '.tiff')
        img_names = [f for f in os.listdir(dir_origin_path)
                     if f.lower().endswith(img_extensions)]

        if not img_names:
            print(f"❌ 输入目录中未找到图片文件")
            exit(1)

        print(f"📄 找到 {len(img_names)} 张图片，开始批量处理...")

        # 批量处理
        success_count = 0
        for img_name in tqdm(img_names, desc="批量检测"):
            try:
                img_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(img_path).convert('RGB')

                # 执行检测
                r_image = yolo.detect_image(image, crop=crop, count=count)

                # 保存结果（保持原格式）
                save_name = img_name
                if save_name.lower().endswith('.jpg'):
                    save_name = save_name.replace('.jpg', '.png')
                save_path = os.path.join(dir_save_path, save_name)
                r_image.save(save_path, quality=95, subsampling=0)

                success_count += 1

            except Exception as e:
                print(f"\n⚠️  处理 {img_name} 失败: {e}")
                continue

        print(f"\n✅ 批量处理完成:")
        print(f"成功: {success_count}/{len(img_names)}")
        print(f"结果保存至: {dir_save_path}")

    # -------------------------- 热力图模式 --------------------------
    elif mode == "heatmap":
        print("\n🔥 热力图可视化模式")
        try:
            # 这里需要根据YOLO类的实际实现调整
            if hasattr(yolo, 'generate_heatmap'):
                yolo.generate_heatmap(save_path=heatmap_save_path)
                print(f"✅ 热力图已保存至: {heatmap_save_path}")
            else:
                print("❌ YOLO模型不支持热力图生成")
        except Exception as e:
            print(f"❌ 生成热力图失败: {e}")

    # -------------------------- ONNX导出模式 --------------------------
    elif mode == "export_onnx":
        print("\n📦 ONNX模型导出模式")
        try:
            # 这里需要根据YOLO类的实际实现调整
            if hasattr(yolo, 'export_onnx'):
                yolo.export_onnx(
                    save_path=onnx_save_path,
                    simplify=simplify,
                    opset_version=12
                )
                print(f"✅ ONNX模型已保存至: {onnx_save_path}")
            else:
                # 兼容ultralytics YOLO的导出方式
                if hasattr(yolo, 'model') and hasattr(yolo.model, 'export'):
                    yolo.model.export(
                        format='onnx',
                        simplify=simplify,
                        opset=12,
                        imgsz=opt.shape,
                        save=onnx_save_path
                    )
                    print(f"✅ ONNX模型已保存至: {onnx_save_path}")
                else:
                    print("❌ YOLO模型不支持ONNX导出")
        except Exception as e:
            print(f"❌ 导出ONNX失败: {e}")

    else:
        print(f"❌ 不支持的模式: {mode}")
        print("支持的模式: dir_predict, video, fps, predict, heatmap, export_onnx")
        exit(1)