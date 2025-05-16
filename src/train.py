import os
from typing import Dict, List


import hydra
from omegaconf import DictConfig
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from src.utils.DatasetLoader import DatasetEnum, get_dataset
import torch, torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from src.models.Res18BaseModel import ResBase18Model
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import Dataset
from torch import nn


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    # print(f"pwd = {os.getcwd()}")
    num_epoch: int = cfg.epoch.num_epoch
    train_id: str = cfg.train_id

    save_root: str = f"./record/{train_id}_{datetime.now().strftime(r'%y%m%d_%H%M')}"
    dataset_name: str = cfg.train.dataset
    start_epoch: int = cfg.train.start_epoch

    # 加载数据集
    dataset: Dict[str, Dataset] = get_dataset(DatasetEnum.str_to_enum(dataset_name))
    dataLoader: Dict[str, DataLoader] = {
        "train": DataLoader(dataset["train"], batch_size=64, shuffle=True, num_workers=8),
        "val": DataLoader(dataset["val"], batch_size=64, shuffle=True, num_workers=8),
    }

    # device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 模型
    model: nn.Module = ResBase18Model().to(device)

    # 冻结指定参数
    # 解冻layer4
    for name, param in model.named_parameters():
        if name.startswith("resnet") and not name.startswith("resnet.layer4"):
            param.requires_grad = False
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)

    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    loss_epoch: Dict[str, List[float]] = {"train": [], "val": []}
    acc_epoch: Dict[str, List[float]] = {"train": [], "val": []}

    with SummaryWriter(os.path.join(save_root, "tb"), filename_suffix=f"ResBase-{datetime.now()}") as writer:

        for epoch in range(start_epoch, num_epoch + 1):
            # 训练
            loss_batch = []
            acc_batch = []
            model.train()
            for batch in tqdm(dataLoader["train"], desc=f"Epoch:{epoch}/{num_epoch}"):
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

                save_checkpoint(save_root, model, loss_epoch, acc_epoch, epoch)

            # writer.add_scalar("loss/train",loss_epoch["train"][-1],epoch)
            # writer.add_scalar("loss/val",loss_epoch["val"][-1],epoch)
            # writer.add_scalar("acc/train",acc_epoch["train"][-1],epoch)
            # writer.add_scalar("acc/val",acc_epoch["val"][-1],epoch)
            writer.add_scalars(
                "loss", {"train": loss_epoch["train"][-1], "val": loss_epoch["val"][-1]}, epoch
            )
            writer.add_scalars("acc", {"train": acc_epoch["train"][-1], "val": acc_epoch["val"][-1]}, epoch)


def save_checkpoint(
    save_root: str,
    model: nn.Module,
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
    create_save_dir(save_dir, is_override=True)

    # 保存在本地中

    fig.savefig(os.path.join(save_dir, f"loss_acc_{epoch}.png"))
    # 保存模型
    torch.save(model.state_dict(), os.path.join(save_dir, f"model_{epoch}.pth"))


# 创建一个文件夹，检查本地是否有，如果有在文件夹名最后递增追加数字
def create_save_dir(save_root: str, *, is_override: bool = False) -> str:
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    elif not is_override:
        i = 1
        while os.path.exists(save_root + f"_{i}"):
            i += 1
        save_root = save_root + f"_{i}"
        os.makedirs(save_root)
    return save_root


if __name__ == "__main__":
    # train()
    ...
