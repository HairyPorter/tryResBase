import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
from torchvision.models import resnet50
from torchvision import transforms
from torchvision.models.resnet import ResNet50_Weights


# 分类器需要根据数据集而变化
class Classifier(nn.Module):

    def __init__(self, input_num: int, num_classes: int):
        super(Classifier, self).__init__()

        """resnet50的fc('fc', Linear(in_features=2048, out_features=1000, bias=True)"""

        self.fc1 = nn.Linear(input_num, num_classes)
        self.bn1 = nn.BatchNorm1d(num_classes)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(num_classes, num_classes)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = self.fc1(x)
        # x = self.softmax(x)
        # 改用一层隐藏层的MLP
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


class ResBase50Model(nn.Module):
    def __init__(self, *, pretrained: bool = False):
        super(ResBase50Model, self).__init__()
        if pretrained:
            self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            self.resnet = resnet50(weights=None)
        # self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()  # type: ignore
        self.classifier = Classifier(2048, 10)

    def forward(self, x):
        x = self.resnet(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":

    model = resnet50()
