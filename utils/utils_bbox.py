import torch
import torch.nn as nn
from torchvision.ops import nms
import numpy as np
from typing import List, Tuple, Optional, Union


class DecodeBox(nn.Module):
    """YOLOv8预测框解码类，负责将模型输出转换为实际坐标并进行后处理"""

    def __init__(self,
                 anchors: List[Tuple[float, float]],
                 num_classes: int,
                 input_shape: Tuple[int, int],
                 anchors_mask: List[List[int]] = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]):
        super(DecodeBox, self).__init__()
        self.anchors = anchors  # 锚框尺寸列表 [(w1,h1), (w2,h2), ...]
        self.num_classes = num_classes  # 类别数量
        self.bbox_attrs = 5 + num_classes  # 每个框的属性数(xywh+conf+classes)
        self.input_shape = input_shape  # 模型输入尺寸 (h, w)
        self.anchors_mask = anchors_mask  # 不同尺度对应的锚框索引

    def decode_box(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        对模型输出进行解码，将预测值转换为相对坐标

        Args:
            inputs: 模型输出的三个尺度特征图，shape分别为
                   [batch, 3*(5+num_classes), 13, 13]
                   [batch, 3*(5+num_classes), 26, 26]
                   [batch, 3*(5+num_classes), 52, 52]

        Returns:
            解码后的预测框列表，每个元素shape为[batch, num_anchors, 5+num_classes]
        """
        outputs = []
        for i, input_tensor in enumerate(inputs):
            batch_size = input_tensor.size(0)
            input_height = input_tensor.size(2)
            input_width = input_tensor.size(3)

            # 计算步长（特征图到输入图像的缩放比例）
            stride_h = self.input_shape[0] / input_height
            stride_w = self.input_shape[1] / input_width

            # 计算特征图尺度下的锚框尺寸
            scaled_anchors = [
                (anchor_w / stride_w, anchor_h / stride_h)
                for anchor_w, anchor_h in [self.anchors[idx] for idx in self.anchors_mask[i]]
            ]

            # 调整输出格式 [batch, 3, h, w, 5+num_classes]
            prediction = input_tensor.view(
                batch_size, len(self.anchors_mask[i]),
                self.bbox_attrs, input_height, input_width
            ).permute(0, 1, 3, 4, 2).contiguous()

            # 解析预测值
            x = torch.sigmoid(prediction[..., 0])  # 中心x坐标偏移量
            y = torch.sigmoid(prediction[..., 1])  # 中心y坐标偏移量
            w = prediction[..., 2]  # 宽度缩放因子
            h = prediction[..., 3]  # 高度缩放因子
            conf = torch.sigmoid(prediction[..., 4])  # 目标置信度
            pred_cls = torch.sigmoid(prediction[..., 5:])  # 类别置信度

            # 设备类型统一
            device = x.device
            FloatTensor = torch.FloatTensor if device.type == 'cpu' else torch.cuda.FloatTensor
            LongTensor = torch.LongTensor if device.type == 'cpu' else torch.cuda.LongTensor

            # 生成网格坐标
            grid_x = torch.linspace(0, input_width - 1, input_width, device=device)
            grid_x = grid_x.repeat(input_height, 1).repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1
            ).view(x.shape).type(FloatTensor)

            grid_y = torch.linspace(0, input_height - 1, input_height, device=device)
            grid_y = grid_y.repeat(input_width, 1).t().repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1
            ).view(y.shape).type(FloatTensor)

            # 生成锚框宽高矩阵
            anchor_w = FloatTensor(scaled_anchors, device=device).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors, device=device).index_select(1, LongTensor([1]))
            anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
            anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

            # 计算预测框坐标（特征图尺度）
            pred_boxes = torch.zeros_like(prediction[..., :4], device=device)
            pred_boxes[..., 0] = x + grid_x  # 中心x
            pred_boxes[..., 1] = y + grid_y  # 中心y
            pred_boxes[..., 2] = torch.exp(w) * anchor_w  # 宽度
            pred_boxes[..., 3] = torch.exp(h) * anchor_h  # 高度

            # 归一化坐标并拼接结果
            scale = torch.tensor([input_width, input_height, input_width, input_height],
                                 device=device, dtype=torch.float32)
            output = torch.cat([
                pred_boxes.view(batch_size, -1, 4) / scale,
                conf.view(batch_size, -1, 1),
                pred_cls.view(batch_size, -1, self.num_classes)
            ], dim=-1)
            outputs.append(output)

        return outputs

    def yolo_correct_boxes(self,
                           box_xy: np.ndarray,
                           box_wh: np.ndarray,
                           input_shape: Tuple[int, int],
                           image_shape: Tuple[int, int],
                           letterbox_image: bool) -> np.ndarray:
        """
        将归一化坐标转换为原始图像坐标

        Args:
            box_xy: 中心坐标 [n, 2]
            box_wh: 宽高 [n, 2]
            input_shape: 模型输入尺寸
            image_shape: 原始图像尺寸
            letterbox_image: 是否使用letterbox缩放

        Returns:
            转换后的坐标 [n, 4]，格式为[x1, y1, x2, y2]
        """
        # 交换xy为yx，适应图像坐标系（y轴向下为正）
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape_np = np.array(input_shape, dtype=np.float32)
        image_shape_np = np.array(image_shape, dtype=np.float32)

        if letterbox_image:
            # 计算letterbox缩放的补偿
            new_shape = np.round(image_shape_np * np.min(input_shape_np / image_shape_np))
            offset = (input_shape_np - new_shape) / 2. / input_shape_np
            scale = input_shape_np / new_shape

            box_yx = (box_yx - offset) * scale
            box_hw *= scale

        # 转换为左上角和右下角坐标
        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([
            box_mins[..., 0:1],  # x1
            box_mins[..., 1:2],  # y1
            box_maxes[..., 0:1],  # x2
            box_maxes[..., 1:2]  # y2
        ], axis=-1)

        # 缩放至原始图像尺寸
        boxes *= np.concatenate([image_shape_np, image_shape_np])
        return boxes

    def non_max_suppression(self,
                            prediction: torch.Tensor,
                            num_classes: int,
                            input_shape: Tuple[int, int],
                            image_shape: Tuple[int, int],
                            letterbox_image: bool,
                            conf_thres: float = 0.5,
                            nms_thres: float = 0.4) -> List[Optional[np.ndarray]]:
        """
        非极大值抑制，过滤重叠框

        Args:
            prediction: 解码后的预测框 [batch, num_anchors, 5+num_classes]
            num_classes: 类别数量
            input_shape: 模型输入尺寸
            image_shape: 原始图像尺寸
            letterbox_image: 是否使用letterbox缩放
            conf_thres: 置信度阈值
            nms_thres: NMS阈值

        Returns:
            过滤后的预测框列表，每个元素为[num_boxes, 7]，格式为[x1,y1,x2,y2,obj_conf,cls_conf,cls_id]
        """
        # 转换为左上角右下角坐标格式
        box_corner = torch.zeros_like(prediction[..., :4])
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2  # x1
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2  # y1
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2  # x2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2  # y2
        prediction[:, :, :4] = box_corner

        output: List[Optional[np.ndarray]] = [None for _ in range(len(prediction))]

        for i, image_pred in enumerate(prediction):
            # 获取每个框的最高类别置信度和类别ID
            class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], dim=1, keepdim=True)

            # 置信度筛选 (目标置信度 * 类别置信度)
            conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]

            if not image_pred.size(0):
                continue  # 没有符合条件的框

            # 拼接检测结果 [x1,y1,x2,y2,obj_conf,cls_conf,cls_id]
            detections = torch.cat([
                image_pred[:, :5],
                class_conf.float(),
                class_pred.float()
            ], dim=1)

            # 按类别进行NMS
            unique_labels = detections[:, -1].cpu().unique()
            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()
                detections = detections.cuda()

            for c in unique_labels:
                # 筛选当前类别的检测框
                class_mask = detections[:, -1] == c
                detections_class = detections[class_mask]

                # 执行非极大值抑制
                keep = nms(
                    detections_class[:, :4],
                    detections_class[:, 4] * detections_class[:, 5],  # 综合置信度
                    nms_thres
                )
                max_detections = detections_class[keep]

                # 积累当前类别的检测结果
                if output[i] is None:
                    output[i] = max_detections
                else:
                    output[i] = torch.cat((output[i], max_detections))

            # 坐标转换并转换为numpy数组
            if output[i] is not None:
                output[i] = output[i].cpu().numpy()
                # 计算中心坐标和宽高
                box_xy = (output[i][:, [0, 1]] + output[i][:, [2, 3]]) / 2  # 中心(x,y)
                box_wh = output[i][:, [2, 3]] - output[i][:, [0, 1]]  # 宽高(w,h)
                # 校正坐标到原始图像
                output[i][:, :4] = self.yolo_correct_boxes(
                    box_xy, box_wh, input_shape, image_shape, letterbox_image
                )

        return output