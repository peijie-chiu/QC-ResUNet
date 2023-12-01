import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet import generate_model

def init_subpixel(weight):
    co, ci, h, w = weight.shape
    co2 = co // 4
    # initialize sub kernel
    k = torch.empty([co2, ci, h, w])
    nn.init.kaiming_uniform_(k)
    # repeat 4 times
    k = k.repeat_interleave(4, dim=0)
    weight.data.copy_(k)

def init_linear(m, relu=True):
    if relu: nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
    else: nn.init.xavier_uniform_(m.weight)
    if m.bias is not None: nn.init.zeros_(m.bias)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, act=True):
        padding = (kernel_size - 1) // 2
        layers = [
          nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
          nn.InstanceNorm3d(out_channels)
        ]
        if act: layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)
    
    def reset_parameters(self):
        init_linear(self[0])
        self[1].reset_parameters()


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

class UpsampleShuffle(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv3d(in_channels, out_channels * 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2)
        )
        
    def reset_parameters(self):
        init_subpixel(self[0].weight)
        nn.init.zeros_(self[0].bias)


class UpsampleBilinear(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        )
    
    def reset_parameters(self):
        init_linear(self[0])


class UpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_t = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        return self.conv_t(x)
    
    def reset_parameters(self):
        init_linear(self.conv_t)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, upsample, skip=True):
        super().__init__()
        if skip:
            self.up = upsample(in_channels, in_channels // 2)
        else:
            self.up = upsample(in_channels, in_channels)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)
        else:
            return self.conv(x1)


class ResUNet(nn.Module):
    def __init__(self, 
                 network_depth, 
                 num_input_channels, 
                 n_classes,
                 out_channels, 
                 upsample=UpsampleBilinear, 
                 skip=True,
                 drop_path=0.0,
                 dropout=0.0):
        super().__init__()
        self.skip = skip

        self.resnet_encoder = generate_model(network_depth, 
                                             num_input_channels=num_input_channels, 
                                             n_classes=n_classes, 
                                             conv1_t_stride=2, 
                                             feature_extarctor=True,
                                             drop_path=drop_path)
        if skip:
            self.up1 = Up(256, 128, upsample)
            self.up2 = Up(128, 128, upsample)
            self.up3 = Up(128, 64, upsample)
        else:
            self.up1 = nn.Conv3d(512, 128, 3, stride=1, padding=12, dilation=12, bias=True)
            self.up2 = nn.Conv3d(128, 2, 3, stride=1, padding=12, dilation=12, bias=True)
            # self.up3 = Up(128, 64, upsample, False)
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, n_classes))
        self.seg_out = nn.Conv3d(64, out_channels, 1)

    def forward(self, x):
        input_size = x.size()
        if self.skip:
            x_reg, cahce_dict = self.resnet_encoder(x, intermediate=True)

            # print(cahce_dict[-1].size(), cahce_dict[-2].size())

            x = self.up1(cahce_dict[-1], cahce_dict[-2])
            x = self.up2(x, cahce_dict[-3])
            x = self.up3(x, cahce_dict[-4]) 
            x = F.interpolate(x, scale_factor=2, mode='trilinear')
            x_seg = self.seg_out(x)
        else:
            x_reg = self.resnet_encoder(x, intermediate=False)
            x = self.up1(x_reg)
            x = self.up2(x)
            x_seg = F.interpolate(x, tuple(list(input_size[-3:])), mode='trilinear')

        x_reg = F.adaptive_avg_pool3d(x_reg, 1).flatten(1)
        x_reg = self.fc(x_reg)

        return x_reg, x_seg

def resunet34(num_input_channels=5, **kwargs):
    return ResUNet(34, num_input_channels, 1, 1, **kwargs)

def resunet50(num_input_channels=5, **kwargs):
    return ResUNet(50, num_input_channels, 1, 1, **kwargs)

if __name__ == "__main__":
    # def clever_format(nums, format="%.2f"):
    #     clever_nums = []

    #     for num in nums:
    #         if num > 1e12:
    #             clever_nums.append(format % (num / 1024 ** 4) + "T")
    #         elif num > 1e9:
    #             clever_nums.append(format % (num / 1024 ** 3) + "G")
    #         elif num > 1e6:
    #             clever_nums.append(format % (num / 1024 ** 2) + "M")
    #         elif num > 1e3:
    #             clever_nums.append(format % (num / 1024) + "K")
    #         else:
    #             clever_nums.append(format % num + "B")

    #     clever_nums = clever_nums[0] if len(clever_nums) == 1 else (*clever_nums, )

    #     return clever_nums
    # from thop import profile
    model = resunet34()
    input = torch.randn(1, 5, 160, 192, 160)
    model(input)
    # model.train()
    # flops, params = profile(model, inputs=(input, ))
    # # print("flops = ", flops)
    # # print("params = ", params)
    # flops, params = clever_format([flops, params], "%.3f")
    # print("flops = ", flops)
    # print("params = ", params)

    # from fvcore.nn import FlopCountAnalysis
    # flops = FlopCountAnalysis(model, input)

    # print(clever_format([flops.total()))

# y = torch.randint(0, 3, (2, 1, 160, 192, 160))
# print(to_onehot(y, 4).size())

# model = ResUNet(34, 5, 3, 1)
# model(torch.randn(2, 5, 160, 192, 160))
