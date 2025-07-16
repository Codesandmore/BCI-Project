import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    def __init__(
        self,
        n_channels=60,
        n_samples=500,
        n_classes=4,
        F1=8,
        D=2,
        F2=16,
        dropout_rate=0.25,
    ):
        super(EEGNet, self).__init__()
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.conv2 = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = None
        self.n_classes = n_classes

    def _set_classifier(self, x):
        in_features = x.shape[1]
        self.classifier = nn.Linear(in_features, self.n_classes).to(x.device)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.avg_pool2d(x, (1, 4))
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = F.avg_pool2d(x, (1, 8))
        x = self.dropout(x)
        x = x.flatten(start_dim=1)
        if self.classifier is None:
            self._set_classifier(x)
        x = self.classifier(x)
        return x

def get_model(n_channels, n_samples, n_classes):
    return EEGNet(n_channels=n_channels, n_samples=n_samples, n_classes=n_classes)



if __name__ == "__main__":
    model = EEGNet(n_channels=60, n_samples=1000, n_classes=4)
    print(model)
    x = torch.randn(8, 60, 1000)
    y = model(x)
    print("Output shape:", y.shape)