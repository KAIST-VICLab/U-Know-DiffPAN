import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------Initialization----------------------------------------
def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):  ## initialization for Conv2d
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # method 2: initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):  ## initialization for BN
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):  ## initialization for nn.Linear
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

# -------------ResNet Block (One)----------------------------------------
class Resblock(nn.Module):
    def __init__(self,in_channels=64,out_channels=64):
        super(Resblock, self).__init__()

        channel = 64
        self.conv20 = nn.Conv2d(in_channels=in_channels, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.conv21 = nn.Conv2d(in_channels=channel, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.conv22 = nn.Conv2d(in_channels=channel, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # x= hp of ms; y = hp of pan
        rs1 = self.relu(self.conv20(x))  # Bsx64x64x64
        rs1 = self.relu(self.conv21(rs1))  # Bsx64x64x64
        rs1 = self.relu(self.conv22(rs1))   # Bsx64x64x64
        rs = torch.add(x, rs1)  # Bsx64x64x64
        return rs

# -----------------------------------------------------
class PanNet_variance(nn.Module):
    def __init__(self, spectral_num, criterion=None, channel=64, reg=True):
        super().__init__()
        self.criterion = criterion
        self.reg = reg

        self.conv1 = nn.Conv2d(in_channels=spectral_num + 1, out_channels=channel, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.res1 = Resblock()
        self.res2 = Resblock()
        self.res3 = Resblock()
        self.res4 = Resblock()

        self.conv3 = nn.Conv2d(in_channels=channel, out_channels = 2* spectral_num, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
            self.res1,
            self.res2,
            self.res3,
            self.res4
        )
        self.spectral_num = spectral_num
        self.apply(init_weights)

    def forward(self, lms, pan):# x= hp of ms; y = hp of pan
        input = torch.cat([lms, pan], 1)  # Bsx9x64x64
        rs = self.relu(self.conv1(input))  # Bsx32x64x64
        rs = self.backbone(rs)  # ResNet's backbone!
        output_variance =  self.conv3(rs)  # Bsx8x64x64
        out = output_variance[:, : -self.spectral_num, ...] + lms
        variance = F.softplus(output_variance[:, -self.spectral_num :, ...])
        output = {'out':out, 'variance':variance}
        return output