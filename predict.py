import time
import yaml
import cv2
import numpy as np
from PIL import Image
from get_yaml import get_config
from yolo import YOLO
import argparse

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='model_data/yolov8n.pt', help='初始权重路径')
    parser.add_argument('--tiny', action='store_true', help='使用yolov8n模型(轻量化模型)')
    parser.add_argument('--phi', type=int, default=1, help='注意力机制类型')
    parser.add_argument('--mode', type=str,
                        choices=['dir_predict', 'video', 'fps', 'predict', 'heatmap', 'export_onnx'],
                        default="dir_predict", help='预测的模式')
    parser.add_argument('--cuda', action='store_true', help='表示是否使用GPU')
    parser.add_argument('--shape', type=int, default=640, help='输入图像的shape')
    parser.add_argument('--video', type=str, default='', help='需要检测的视频文件')
    parser.add_argument('--save-video', type=str, default='', help='保存视频的位置')
    parser.add_argument('--confidence', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--nms_iou', type=float, default=0.3, help='非极大抑制阈值')
    opt = parser.parse_args()

    # 加载配置和模型
    config = get_config()
    yolo = YOLO(opt)

    # 根据不同模式执行不同操作
    mode = opt.mode
    crop = False  # 是否裁剪目标
    count = False  # 是否计数

    # 视频检测相关参数
    video_path = 0 if opt.video == '' else opt.video
    video_save_path = opt.save_video
    video_fps = 25.0

    # FPS测试相关参数
    test_interval = 100
    fps_image_path = "img/call.jpg"

    # 目录检测相关参数
    dir_origin_path = "img/"
    dir_save_path = "img_out/"

    # 热力图相关参数
    heatmap_save_path = "model_data/heatmap_vision.png"

    # ONNX导出相关参数
    simplify = True
    onnx_save_path = "model_data/models.onnx"

    # 单张图片预测模式
    if mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, crop=crop, count=count)
                r_image.show()
                r_image.save(dir_save_path + 'img_result.jpg')

    # 视频检测模式
    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path != '':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频）")

        fps = 0.0
        while (True):
            t1 = time.time()
            # 读取帧并转换格式
            ref, frame = capture.read()
            if not ref:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            # 检测
            frame = np.array(yolo.detect_image(frame))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 计算FPS
            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != '':
                out.write(frame)

            if c == 27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path != '':
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    # FPS测试模式
    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    # 目录检测模式
    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                r_image = yolo.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

