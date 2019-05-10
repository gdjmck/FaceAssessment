import torch
import torch.nn as nn
import torch.nn.functional as F
import extractors

class SeparableBlock(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size=3, stride=1, padding=1):
        super(SeparableBlock, self).__init__()
        self.depth_wise_conv = nn.Conv2d(in_chan, in_chan, kernel_size=kernel_size, stride=stride padding=padding, groups=in_chan)
        self.pt_wise_conv = nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1)
        self.op = nn.Sequential(self.depth_wise_conv, 
                                self.pt_wise_conv, 
                                nn.BatchNorm2d(out_chan), 
                                nn.ReLU(inplace=True))

    def forward(self, x):
        return self.op(x)

class FaceAssess(nn.Module):
    def __init__(self):
        super(FaceAssess, self).__init__()
        self.extractor = extractors.resnet50(pretrained=True)
        self.conv1 = nn.Sequential(nn.Conv2d(2048, 1024, 1), 
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=(2, 2)))
        self.sep_squeeze_1 = SeparableBlock(1024, 512)
        self.sep_compact_1 = SeparableBlock(512, 512, stride=2)
        self.sep_squeeze_2 = SeparableBlock(512, 256)
        self.sep_compact_2 = SeparableBlock(256, 256, stride=2)
        self.sep_squeeze_3 = SeparableBlock(256, 128)
        self.sep_compact_3 = SeparableBlock(128, 128, stride=2)
        self.sep_squeeze_4 = SeparableBlock(128, 64)
        self.linear = nn.Linear(64, 1)

    def forward(self, x):
        feat = self.extractor(x)[0] # -> (-1, 2048, 1/8, 1/8)
        # 对feat做ROI pooling到 (-1, -1, 32, 32)
        feat = F.adaptive_max_pool2d(feat, output_size=(32, 32))
        feat = self.conv1(feat) # -> (-1, 1024, 1/16, 1/16)  16x16
        feat = self.sep_squeeze_1(feat) # -> (-1, 512, 1/16, 1/16)  16x16
        feat = self.sep_compact_1(feat) # -> (-1, 512, 1/32, 1/32)  8x8
        feat = self.sep_squeeze_2(feat) # -> (-1, 256, 1/32, 1/32)  8x8
        feat = self.sep_compact_2(feat) # -> (-1, 256, 1/64, 1/64)  4x4
        feat = self.sep_squeeze_3(feat) # -> (-1, 128, 1/64, 1/64)  4x4
        feat = self.sep_compact_3(feat) # -> (-1, 128, 1/128, 1/128)  2x2
        feat = self.sep_squeeze_4(feat) # -> (-1, 64, 1/128, 1/128)  2x2
        feat = F.max_pool2d(feat, kernel_size=2)
        feat = feat.reshape((-1,))
        score = self.linear(feat)
        score = F.sigmoid(score)
        # 让一些Layer负责监测有没有人脸，另外的做区域模糊监测，或者只对人脸检测层加regularization
        return score