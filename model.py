import torch
import torch.nn as nn
import extractors

class FaceAssess(nn.Module):
    def __init__(self):
        super(self, FaceAssess).__init__()
        self.extractor = extractors.resnet50(pretrained=True)
        self.conv1 = nn.Sequential([nn.Conv2d(2048, 1024, 1), 
                                    nn.BatchNorm2d(1024)])

    def forward(self, x):
        feat = self.extractor(x) # (-1, 2048, 1/8, 1/8)