

import os
from typing import Dict

from numpy import save
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from DatasetLoader import DatasetEnum, get_dataset
import torch,torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from ResBaseModel import ResBaseModel
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import Dataset

if __name__ == "__main__":
    # 加载数据集    
    num_epoch = 100
    comment = "ResBase_1"
    save_root = f"./record/{comment}_{datetime.now().strftime(r'%y%m%d_%H%M')}"

    

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    dataset_name = DatasetEnum.MNIST

    dataset:Dict[str,Dataset] = get_dataset(dataset_name)
    dataLoader = {
        "train": DataLoader(dataset["train"], batch_size=64, shuffle=True,num_workers=8),
        "val": DataLoader(dataset["val"], batch_size=64, shuffle=True,num_workers=8),
    }
    # for batch in dataLoader["train"]:
    #     print(batch[0].shape)
    #     break
    # for batch in dataLoader["val"]:
    #     print(batch[0].shape)
    #     break
    model = ResBaseModel().to(device)
    # 冻结指定参数
    for name, param in model.named_parameters():
        if name.startswith("resnet"):
            param.requires_grad = False
    optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad, model.parameters()), lr=0.01)

    loss_fn = torch.nn.CrossEntropyLoss().to(device)




    loss_epoch = {"train": [], "val": []}
    acc_epoch = {"train": [], "val":[]}



    with SummaryWriter(os.path.join(save_root,"tb"),filename_suffix=f"ResBase-{datetime.now()}") as writer:
    
        for epoch in range(1,num_epoch+1):
            # 训练
            loss_batch = []
            acc_batch = []
            model.train()
            for batch in tqdm(dataLoader["train"],desc=f"Epoch:{epoch}/{num_epoch}"):
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

                fig,axs = plt.subplots(1,2)
                axs[0].plot(loss_epoch["train"],label="train")
                axs[0].plot(loss_epoch["val"],label="val")
                axs[0].set_title("loss")
                axs[0].set_xlabel("epoch")
                axs[0].legend()
                axs[1].plot(acc_epoch["train"],label="train")
                axs[1].plot(acc_epoch["val"],label="val")
                axs[1].set_title("acc")
                axs[1].set_xlabel("epoch")
                axs[1].legend()
                # 创建文件夹
                save_dir = os.path.join(save_root,f"epoch_{epoch}")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # 保存在本地中
                
                fig.savefig(os.path.join(save_dir,f"loss_acc_{epoch}.png"))
                # 保存模型
                torch.save(model.state_dict(),os.path.join(save_dir,f"model_{epoch}.pth"))

                
            # writer.add_scalar("loss/train",loss_epoch["train"][-1],epoch)
            # writer.add_scalar("loss/val",loss_epoch["val"][-1],epoch)
            # writer.add_scalar("acc/train",acc_epoch["train"][-1],epoch)
            # writer.add_scalar("acc/val",acc_epoch["val"][-1],epoch)
            writer.add_scalars("loss",{"train":loss_epoch["train"][-1],"val":loss_epoch["val"][-1]},epoch)
            writer.add_scalars("acc",{"train":acc_epoch["train"][-1],"val":acc_epoch["val"][-1]},epoch)



        