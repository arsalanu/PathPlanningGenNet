import torch
import torch.nn as nn
import torch.utils as utils
import numpy as np

class PathGen(nn.Module):
    def __init__(self):
        super(PathGen, self).__init__()
        
        self.Conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=(3,3),
                stride=2,
                padding=1,
                padding_mode='zeros'
            ),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.Conv_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3,3),
                stride=2,
                padding=1,
                padding_mode='zeros'
            ),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.Conv_R3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(3,3),
                stride=1,
                padding=1,
                padding_mode='zeros'
            ),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.Conv_4 = nn.Sequential(
            nn.Conv2d(
                in_channels = 32,
                out_channels = 64,
                kernel_size=(3,3),
                stride=2,
                padding=1,
                padding_mode='zeros'
            ),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.DeConv_1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=(4,4),
                stride=2,
                padding=1,
                padding_mode='zeros'
            ),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.5),
            nn.ReLU()
        )

        self.DeConv_R2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(3,3),
                stride=1,
                padding=1,
                padding_mode='zeros'
            ),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.DeConv_3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=(4,4),
                stride=2,
                padding=1,
                padding_mode='zeros'
            ),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        self.DeConv_4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=1,
                kernel_size=(4,4),
                stride=2,
                padding=1,
                padding_mode='zeros'
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.Conv_1(x)

        x = self.Conv_2(x)

        for _ in range(0):
            x = self.Conv_R3(x)

        x = self.Conv_4(x)

    #------------------------------

        x = self.DeConv_1(x)

        for _ in range(10):
            x = self.DeConv_R2(x)
            
        x = self.DeConv_3(x)

        o = self.DeConv_4(x)
        
        return o