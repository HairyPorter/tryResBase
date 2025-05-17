import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
from torchvision.models import resnet18
from torchvision import transforms
from torchvision.models.resnet import ResNet18_Weights
from torchvision.models.feature_extraction import (
    get_graph_node_names,
    create_feature_extractor,
)


class Classifier(nn.Module):

    def __init__(self, input_num: int, num_classes: int):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_num, num_classes)
        self.fc2 = nn.Linear(num_classes, num_classes)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = self.fc1(x)
        # x = self.softmax(x)
        # 改用一层隐藏层的MLP
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ResBase18Model(nn.Module):
    def __init__(self, *, pretrained: bool = False):
        super(ResBase18Model, self).__init__()
        if pretrained:
            self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            self.resnet = resnet18(weights=None)
        # self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()  # type: ignore
        self.classifier = Classifier(512, 10)

    def forward(self, x):
        x = self.resnet(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    from utils.dataset_loader import DatasetEnum, get_dataset

    # model = ResBaseModel()

    # print(f"{model(torch.randn(1, 3, 224,224)).shape=}")
    # download_dataset("MNIST")
    # get_dataset("MNIST")
    # dataset = get_dataset(DatasetEnum.MNIST)
    # dataLoder = {}
    # dataLoder["train"] = DataLoader(dataset["train"], batch_size=64, shuffle=False)
    # dataLoder["val"] = DataLoader(dataset["val"], batch_size=64, shuffle=False)
    ...
