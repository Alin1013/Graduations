import datetime
import os

import torch
import matplotlib

matplotlib.use('Agg')  # 非交互式环境使用Agg后端
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        # 生成带时间戳的日志目录，确保唯一性
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        self.log_dir = os.path.join(log_dir, f"loss_{time_str}")
        self.losses = []  # 存储训练损失
        self.val_loss = []  # 存储验证损失

        # 初始化日志目录和TensorBoard writer
        os.makedirs(self.log_dir, exist_ok=True)  # exist_ok避免目录已存在的错误
        self.writer = SummaryWriter(self.log_dir)

        # 尝试添加模型结构图到TensorBoard
        try:
            # 生成符合输入形状的虚拟数据 (batch_size=2, 3通道, 高, 宽)
            dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except Exception as e:
            print(f"警告：添加模型图到TensorBoard失败，原因：{e}")

    def append_loss(self, epoch, loss, val_loss):
        # 记录损失值到列表
        self.losses.append(loss)
        self.val_loss.append(val_loss)

        # 写入文本文件（格式化保留6位小数）
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(f"{loss:.6f}\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(f"{val_loss:.6f}\n")

        # 写入TensorBoard
        self.writer.add_scalar('train_loss', loss, epoch)  # 明确命名为train_loss
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()  # 绘制损失曲线

    def loss_plot(self):
        iters = range(len(self.losses))  # 迭代次数（epoch数）

        # 使用上下文管理器自动关闭图片资源
        with plt.figure(figsize=(10, 6)):
            # 绘制原始损失曲线
            plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
            plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')

            # 绘制平滑损失曲线（处理潜在的窗口大小错误）
            try:
                n = len(self.losses)
                # 窗口大小：不超过数据长度的1/5，且为奇数（savgol_filter要求）
                num = min(15, max(3, n // 5))  # 最小3，最大15
                num = num if num % 2 == 1 else num - 1  # 确保为奇数

                if num <= n:  # 窗口大小必须小于等于数据长度
                    # 平滑滤波（窗口大小num，3次多项式拟合）
                    smooth_train = scipy.signal.savgol_filter(self.losses, num, 3)
                    smooth_val = scipy.signal.savgol_filter(self.val_loss, num, 3)
                    plt.plot(iters, smooth_train, 'green', linestyle='--', linewidth=2, label='smooth train loss')
                    plt.plot(iters, smooth_val, '#8B4513', linestyle='--', linewidth=2, label='smooth val loss')
                else:
                    print(f"警告：数据量不足，无法生成平滑曲线（当前数据量：{n}，所需窗口：{num}）")
            except Exception as e:
                print(f"警告：平滑曲线绘制失败，原因：{e}")

            # 图表美化
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.legend(loc="upper right", fontsize=10)
            plt.title('Training and Validation Loss', fontsize=14)

            # 保存图片（高DPI确保清晰度）
            plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"), dpi=300, bbox_inches='tight')

        plt.close('all')  # 确保关闭所有图片，释放资源