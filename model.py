import torch.nn as nn
import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class MyResNet50(nn.Module):
    def __init__(self):
        super(MyResNet50, self).__init__()

        # Load the pre-trained ResNet50 model
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Replace the final fully connected layer
        # The original ResNet50 has 2048 input features to the final layer
        self.model.fc = nn.Linear(2048, 12)

    def forward(self, x):
        return self.model(x)
