import os
from typing import Optional, List, Tuple
import torch
from torch.optim import Optimizer
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast  # 移至顶部导入

from utils.utils import get_lr
from utils.callbacks import LossHistory  # 假设LossHistory的导入路径


def fit_one_epoch(
        model_train: torch.nn.Module,
        model: torch.nn.Module,
        yolo_loss: torch.nn.Module,
        loss_history: LossHistory,
        optimizer: Optimizer,
        epoch: int,
        epoch_step: int,
        epoch_step_val: int,
        gen: List[Tuple[torch.Tensor, List[torch.Tensor]]],
        gen_val: List[Tuple[torch.Tensor, List[torch.Tensor]]],
        Epoch: int,
        cuda: bool,
        fp16: bool,
        scaler: Optional[GradScaler],
        save_period: int,
        save_dir: str,
        local_rank: int = 0
) -> None:
    loss: float = 0.0
    val_loss: float = 0.0

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(
            total=epoch_step,
            desc=f'Epoch {epoch + 1}/{Epoch}',
            postfix=dict,
            mininterval=0.3
        )

    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images = images.cuda(non_blocking=True)  # 添加non_blocking加速
                targets = [ann.cuda(non_blocking=True) for ann in targets]

        # 清零梯度
        optimizer.zero_grad()

        if not fp16:
            # 前向传播
            outputs = model_train(images)
            loss_value_all = 0.0

            # 计算损失
            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all += loss_item
            loss_value = loss_value_all

            # 反向传播
            loss_value.backward()
            optimizer.step()
        else:
            with autocast():
                outputs = model_train(images)
                loss_value_all = 0.0

                for l in range(len(outputs)):
                    loss_item = yolo_loss(l, outputs[l], targets)
                    loss_value_all += loss_item
                loss_value = loss_value_all

            # 混合精度反向传播
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()

        loss += loss_value.item()

        if local_rank == 0:
            pbar.set_postfix(
                **{'loss': loss / (iteration + 1),
                   'lr': get_lr(optimizer)}
            )
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(
            total=epoch_step_val,
            desc=f'Epoch {epoch + 1}/{Epoch}',
            postfix=dict,
            mininterval=0.3
        )

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():  # 验证阶段无需计算梯度
            if cuda:
                images = images.cuda(non_blocking=True)
                targets = [ann.cuda(non_blocking=True) for ann in targets]

            # 前向传播（删除不必要的optimizer.zero_grad()）
            outputs = model_train(images)
            loss_value_all = 0.0

            # 计算损失
            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all += loss_item
            loss_value = loss_value_all

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        print(f'Epoch: {epoch + 1}/{Epoch}')
        print(f'Total Loss: {loss / epoch_step:.3f} || Val Loss: {val_loss / epoch_step_val:.3f}')

        # 按周期保存模型
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            save_path = os.path.join(
                save_dir,
                f"ep{epoch + 1:03d}-loss{loss / epoch_step:.3f}-val_loss{val_loss / epoch_step_val:.3f}.pth"
            )
            torch.save(model.state_dict(), save_path)

        # 保存最新模型
        torch.save(model.state_dict(), os.path.join(save_dir, "last.pth"))