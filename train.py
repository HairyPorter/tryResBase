import os
import shutil
import sys
from typing import Any, Dict, List, Mapping, Optional

import hydra
from omegaconf import DictConfig
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import torch, torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import Dataset
from torch import nn

from src.models.res18base_model import ResBase18Model
from src.utils.dataset_loader import DatasetEnum, get_dataset
from src.utils.file_ops import makedirs_inc_suffix, copydir_inc_suffix


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:

    # 训练开始时间
    start_time: datetime = datetime.now()
    start_time_str: str = start_time.strftime(r"%y%m%d_%H%M")

    num_epoch: int = cfg.train.num_epoch
    train_name: str = cfg.train_name
    train_id: str = f"{train_name}_{start_time_str}"
    # save_root: str = f"./record/{train_id}_{datetime.now().strftime(r'%y%m%d_%H%M')}"
    # 仅依靠 train_id 来确定保存路径，创建时是否覆盖取决与配置文件new_training，
    # 覆盖意味着继续训练，不覆盖意味着新的一轮
    record_dir: str = f"./record/{train_id}"
    dataset_name: str = cfg.train.dataset
    datasets_root: str = cfg.train.datasets_root
    new_training: bool = cfg.train.new_training
    start_epoch: int = cfg.train.last.start_epoch if not new_training else 0
    last_train_id: str = cfg.train.last.id
    last_record_dir: str = f"./record/{last_train_id}"

    # device
    if torch.cuda.is_available():
        device = torch.device("cuda:1")
    else:
        device = torch.device("cpu")

    # record目录
    if not new_training:
        record_dir = copydir_inc_suffix(last_record_dir, record_dir)

    else:
        record_dir = makedirs_inc_suffix(record_dir)

    # checkpoint加载
    checkpoint: Optional[Dict[str, Any]] = None
    if not new_training:
        checkpoint = torch.load(f"{record_dir}/epoch_{start_epoch}/model_{start_epoch}.pth")

    # 模型创建
    if not new_training and checkpoint is not None:
        model: nn.Module = ResBase18Model()

        model.load_state_dict(checkpoint["model"])
    else:
        model: nn.Module = ResBase18Model(pretrained=True)
    model.to(device)

    # 冻结指定参数
    # 解冻layer4
    for name, param in model.named_parameters():
        if name.startswith("resnet") and not name.startswith("resnet.layer4"):
            param.requires_grad = False

    # 优化器传入所有参数, 方便冻结的参数变更
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    if not new_training and checkpoint is not None:
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        optimizer.load_state_dict(checkpoint["optimizer"])

    # 损失函数设置
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # 加载数据集
    dataset: Mapping[str, Dataset] = get_dataset(DatasetEnum.str_to_enum(dataset_name), datasets_root)
    dataLoader: Dict[str, DataLoader] = {
        "train": DataLoader(dataset["train"], batch_size=64, shuffle=True, num_workers=8),
        "val": DataLoader(dataset["val"], batch_size=64, shuffle=True, num_workers=8),
    }

    # 记录指标
    loss_epoch: Dict[str, List[float]] = {"train": [], "val": []}
    acc_epoch: Dict[str, List[float]] = {"train": [], "val": []}

    with SummaryWriter(os.path.join(record_dir, "tb"), filename_suffix=f"{train_id}") as writer:
        # 保存配置文件config.yaml，train.py

        shutil.copyfile("./conf/config.yaml", os.path.join(record_dir, f"{train_id}_config.yaml"))
        shutil.copyfile("./train.py", os.path.join(record_dir, f"{train_id}_train.py"))
        for epoch in range(start_epoch + 1, start_epoch + num_epoch + 1):
            # 训练
            loss_batch = []
            acc_batch = []
            model.train()
            for batch in tqdm(dataLoader["train"], desc=f"Epoch:{epoch}/{start_epoch + num_epoch}"):
                x = batch[0].to(device)
                y = batch[1].to(device)
                optimizer.zero_grad()

                pred = model(x)
                loss_value = loss_fn(pred, y)

                loss_value.backward()
                optimizer.step()

                loss_batch.append(loss_value.item())
                acc_batch.append(torch.mean((torch.argmax(pred, dim=1) == y).float()).item())
            loss_epoch["train"].append(sum(loss_batch) / len(loss_batch))
            acc_epoch["train"].append(sum(acc_batch) / len(acc_batch))

            # 测试
            loss_batch = []
            acc_batch = []
            model.eval()
            for batch in dataLoader["val"]:
                x = batch[0].to(device)
                y = batch[1].to(device)
                pred = model(x)
                acc_batch.append(torch.mean((torch.argmax(pred, dim=1) == y).float()).item())
                loss_batch.append(loss_fn(pred, y).item())
            loss_epoch["val"].append(sum(loss_batch) / len(loss_batch))
            acc_epoch["val"].append(sum(acc_batch) / len(acc_batch))

            if (epoch % 10 == 0 and epoch != 0) or epoch == num_epoch:

                save_checkpoint(record_dir, model, optimizer, loss_epoch, acc_epoch, epoch)

            # writer.add_scalar("loss/train",loss_epoch["train"][-1],epoch)
            # writer.add_scalar("loss/val",loss_epoch["val"][-1],epoch)
            # writer.add_scalar("acc/train",acc_epoch["train"][-1],epoch)
            # writer.add_scalar("acc/val",acc_epoch["val"][-1],epoch)
            writer.add_scalars(
                "loss", {"train": loss_epoch["train"][-1], "val": loss_epoch["val"][-1]}, epoch
            )
            writer.add_scalars("acc", {"train": acc_epoch["train"][-1], "val": acc_epoch["val"][-1]}, epoch)
    print(f"Finished Training.Time spent:{(datetime.now()-start_time)}")


def save_checkpoint(
    save_root: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_epoch: Dict[str, List[float]],
    acc_epoch: Dict[str, List[float]],
    epoch: int,
):
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(loss_epoch["train"], label="train")
    axs[0].plot(loss_epoch["val"], label="val")
    axs[0].set_title("loss")
    axs[0].set_xlabel("epoch")
    axs[0].legend()
    axs[1].plot(acc_epoch["train"], label="train")
    axs[1].plot(acc_epoch["val"], label="val")
    axs[1].set_title("acc")
    axs[1].set_xlabel("epoch")
    axs[1].legend()
    # 创建文件夹
    save_dir = os.path.join(save_root, f"epoch_{epoch}")
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    makedirs_inc_suffix(save_dir, use_existing=True)

    # 保存指标缩略图
    fig.savefig(os.path.join(save_dir, f"loss_acc_{epoch}.png"))
    # 保存模型
    checkpoint: dict = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, os.path.join(save_dir, f"model_{epoch}.pth"))


if __name__ == "__main__":
    train()
    # ...
