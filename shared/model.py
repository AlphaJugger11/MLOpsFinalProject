# shared/model.py
import torch.nn as nn

class GlobalNet(nn.Module):
    def __init__(self, in_dim=561, hidden_layers=[256, 128], num_classes=6):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def get_model(in_dim=561, num_classes=6):
    return GlobalNet(in_dim=in_dim, hidden_layers=[256,128], num_classes=num_classes)
