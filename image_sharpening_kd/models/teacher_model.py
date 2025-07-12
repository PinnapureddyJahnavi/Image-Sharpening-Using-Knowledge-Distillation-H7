
import torch.nn as nn
from torchvision.models import resnet34

class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        base = resnet34(pretrained=True)
        self.model = nn.Sequential(*(list(base.children())[:-2]), nn.Conv2d(512, 3, 3, padding=1))

    def forward(self, x):
        return self.model(x)
