import json
import os
import argparse
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from utils.utils import cvtColor, preprocess_input, resize_image
from yolo import YOLO


def calculate_coco_map(map_mode=0, cocoGt_path='coco_dataset/annotations/instances_val2026.json',
                       dataset_img_path='coco_dataset/val2026', temp_save_path='map_out/coco_eval'):
    """
    计算COCO数据集的mAP
    map_mode: 0-完整流程, 1-仅生成预测结果, 2-仅计算mAP
    """
    # 创建临时目录
    os.makedirs(temp_save_path, exist_ok=True)

    # 加载COCO标注
    try:
        cocoGt = COCO(cocoGt_path)
    except Exception as e:
        raise ValueError(f"加载COCO标注失败: {str(e)}")

    ids = list(cocoGt.imgToAnns.keys())
    clsid2catid = cocoGt.getCatIds()  # 类别ID映射

    # 生成预测结果
    if map_mode == 0 or map_mode == 1:
        yolo = mAP_YOLO(confidence=0.001, nms_iou=0.65)
        results = []

        print("开始生成预测结果...")
        for image_id in tqdm(ids):
            try:
                img_info = cocoGt.loadImgs(image_id)[0]
                image_path = os.path.join(dataset_img_path, img_info['file_name'])
                if not os.path.exists(image_path):
                    print(f"警告: 图像不存在 {image_path}")
                    continue

                image = Image.open(image_path)
                results = yolo.detect_image(image_id, image, results)
            except Exception as e:
                print(f"处理图像 {image_id} 失败: {str(e)}")
                continue

        # 保存预测结果
        with open(os.path.join(temp_save_path, 'eval_results.json'), "w") as f:
            json.dump(results, f)
        print(f"预测结果已保存至 {temp_save_path}/eval_results.json")

    # 计算mAP
    if map_mode == 0 or map_mode == 2:
        print("开始计算mAP...")
        try:
            cocoDt = cocoGt.loadRes(os.path.join(temp_save_path, 'eval_results.json'))
            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            print("mAP计算完成!")

            # 保存评估结果
            with open(os.path.join(temp_save_path, 'map_results.txt'), 'w') as f:
                f.write("COCO mAP评估结果:\n")
                f.write(f"mAP@0.5: {cocoEval.stats[1]:.4f}\n")
                f.write(f"mAP@0.5:0.95: {cocoEval.stats[0]:.4f}\n")
                f.write(f"mAP@0.75: {cocoEval.stats[2]:.4f}\n")
        except Exception as e:
            raise ValueError(f"计算mAP失败: {str(e)}")


class mAP_YOLO(YOLO):
    """用于mAP计算的YOLO子类"""

    def detect_image(self, image_id, image, results):
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data, _ = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            outputs = self.bbox_util.non_max_suppression(
                torch.cat(outputs, 1), self.num_classes, self.input_shape,
                image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou
            )

            if outputs[0] is None:
                return results

            top_label = np.array(outputs[0][:, 6], dtype='int32')
            top_conf = outputs[0][:, 4] * outputs[0][:, 5]
            top_boxes = outputs[0][:, :4]

        # 整理预测结果
        for i, c in enumerate(top_label):
            result = {
                "image_id": int(image_id),
                "category_id": clsid2catid[c],  # 映射到COCO类别ID
                "bbox": [float(top_boxes[i][0]), float(top_boxes[i][1]),
                         float(top_boxes[i][2] - top_boxes[i][0]),
                         float(top_boxes[i][3] - top_boxes[i][1])],
                "score": float(top_conf[i])
            }
            results.append(result)
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='计算COCO数据集mAP')
    parser.add_argument('--map_mode', type=int, default=0,
                        help='0:完整流程, 1:仅生成预测, 2:仅计算mAP')
    parser.add_argument('--coco_gt', default='coco_dataset/annotations/instances_val2026.json',
                        help='COCO验证集标注路径')
    parser.add_argument('--img_path', default='coco_dataset/val2026',
                        help='验证集图像路径')
    parser.add_argument('--save_path', default='map_out/coco_eval',
                        help='结果保存路径')
    args = parser.parse_args()

    calculate_coco_map(
        map_mode=args.map_mode,
        cocoGt_path=args.coco_gt,
        dataset_img_path=args.img_path,
        temp_save_path=args.save_path
    )