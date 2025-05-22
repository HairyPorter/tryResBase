import os
from typing import Dict, Mapping, Tuple
from enum import Enum
import torchvision, torch
from torch.utils.data import Dataset
import torch.utils
import torch.utils.data
from torchvision import transforms
from torchvision.datasets import MNIST, ImageFolder


class DatasetEnum(Enum):

    MNIST = "MNIST"

    @classmethod
    def str_to_enum(cls, name: str) -> "DatasetEnum":
        for member in cls:
            if member.value == name:
                return member
        raise ValueError(f"{name} is not a valid {cls.__name__}")


def get_dataset(dataset_name: DatasetEnum, datasets_root: str) -> Mapping[str, Dataset]:
    """获取数据集字典，包括train和val"""
    dataset_name_list = ["MNIST"]
    if dataset_name.value not in dataset_name_list:
        raise ValueError("Dataset not found")

    if dataset_name == DatasetEnum.MNIST:
        return get_MNIST(datasets_root)


def get_Market1501(dataset_dir: str, size: Tuple[int, int] | int, train_all=False) -> Dict[str, ImageFolder]:
    if isinstance(size, int):
        h = size
        w = size
    elif isinstance(size, tuple):
        h, w = size
    else:
        raise ValueError("size must be int or tuple")
    train_transforms = transforms.Compose(
        [
            # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
            transforms.Resize((h, w), interpolation=3),  # type: ignore
            transforms.Pad(10),
            transforms.RandomCrop((h, w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize(size=(h, w), interpolation=3),  # Image.BICUBIC # type: ignore
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    dataset_dir = os.path.join(dataset_dir, "pytorch")
    image_dataset: Dict[str, ImageFolder] = {}
    image_dataset["train"] = torchvision.datasets.ImageFolder(
        os.path.join(dataset_dir, "train"), train_transforms
    )
    image_dataset["val"] = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, "val"), val_transforms)

    return image_dataset


def get_MNIST(datasets_root: str) -> Dict[str, MNIST]:
    train_transforms = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.RandomRotation(15),
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
    dataset: Dict[str, MNIST] = {}
    dataset["train"] = torchvision.datasets.MNIST(
        root=datasets_root, train=True, transform=train_transforms, download=True
    )
    dataset["val"] = torchvision.datasets.MNIST(
        root=datasets_root, train=False, transform=val_transforms, download=True
    )
    return dataset
