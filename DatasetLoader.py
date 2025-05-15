import torch.utils
import torch.utils.data
import torchvision, torch
from torchvision import transforms
from typing import Dict
from enum import Enum


class DatasetEnum(Enum):
    MNIST = "MNIST"



def get_dataset(dataset_name: DatasetEnum) -> Dict[str, torch.utils.data.Dataset]:
    dataset_name_list = ["MNIST"]
    if dataset_name.value not in dataset_name_list:
        raise ValueError("Dataset not found")
    
    if dataset_name == DatasetEnum.MNIST:
        return get_MNIST()
    

def get_MNIST() -> Dict[str, torch.utils.data.Dataset]:
    train_transforms = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transforms = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = {}
    dataset["train"] = torchvision.datasets.MNIST(root="../datasets", train=True,transform=train_transforms, download=True)
    dataset["val"] = torchvision.datasets.MNIST(root="../datasets", train=False,transform=val_transforms, download=True)
    return dataset