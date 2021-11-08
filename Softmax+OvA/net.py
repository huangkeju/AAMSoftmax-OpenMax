import torch.nn as nn
from torch.nn.modules.linear import Linear

class Net(nn.Module):
    def __init__(self, class_num, feat_dim, dropout):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.bn1 = nn.BatchNorm2d(10)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 10, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(10, 10, 3, padding=1)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(7840, feat_dim)
        self.ova_clf = nn.Sequential(
            nn.Linear(feat_dim, class_num),
            # nn.ReLU(),
            # nn.Linear(class_num, class_num),
        )

    def forward(self, inputs):
        out0 = self.conv1(inputs)
        out0 = self.bn1(out0)
        out0 = self.relu1(out0)
        out = self.conv2(out0)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out += out0
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        pred = self.ova_clf(out)
        return out, pred