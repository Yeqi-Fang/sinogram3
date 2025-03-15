import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=False, attention=False, pretrain=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.attention = attention
        self.pretrain = pretrain
        factor = 2 if bilinear else 1
        if pretrain:
            # 使用预训练的 ResNet 作为编码器
            resnet = models.resnet34(pretrained=True)
            # 修改第一层以接受自定义通道数
            if n_channels != 3:
                self.input_layer = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.input_layer.weight.data = torch.mean(resnet.conv1.weight.data, dim=1, keepdim=True).repeat(1, n_channels, 1, 1)
            else:
                self.input_layer = resnet.conv1
                
            # ResNet 编码器层
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.maxpool = resnet.maxpool
            self.layer1 = resnet.layer1  # 64 通道
            self.layer2 = resnet.layer2  # 128 通道
            self.layer3 = resnet.layer3  # 256 通道
            self.layer4 = resnet.layer4  # 512 通道

            # 为 ResNet 调整解码器通道
            if self.attention:
                self.up1 = AttentionUp(512, 256 // factor, bilinear)
                self.up2 = AttentionUp(256, 128 // factor, bilinear)
                self.up3 = AttentionUp(128, 64 // factor, bilinear)
                self.up4 = AttentionUp(64, 64, bilinear)
            else:
                self.up1 = Up(512, 256 // factor, bilinear)
                self.up2 = Up(256, 128 // factor, bilinear)
                self.up3 = Up(128, 64 // factor, bilinear)
                self.up4 = Up(64, 64, bilinear)

        else:
            self.inc = DoubleConv(n_channels, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            self.down4 = Down(512, 1024 // factor)
            if self.attention:
                self.up1 = AttentionUp(1024, 512 // factor, bilinear)
                self.up2 = AttentionUp(512, 256 // factor, bilinear)
                self.up3 = AttentionUp(256, 128 // factor, bilinear)
                self.up4 = AttentionUp(128, 64, bilinear)
            else:
                self.up1 = Up(1024, 512 // factor, bilinear)
                self.up2 = Up(512, 256 // factor, bilinear)
                self.up3 = Up(256, 128 // factor, bilinear)
                self.up4 = Up(128, 64, bilinear)
                
                
        self.outc = OutConv(64, n_classes)
        # Residual connection (adding the input to the output)
        self.use_residual = True

    def forward(self, x):
        x_in = x  # Store input for potential residual connection
        if self.pretrain:
            # 通过预训练编码器前向传播
            x = self.input_layer(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            x1 = x  # 第一个特征图
            x2 = self.layer1(x1)
            x3 = self.layer2(x2)
            x4 = self.layer3(x3)
            x5 = self.layer4(x4)
        else:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            
            
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        
        # 注意：当使用预训练模型时，残差连接可能需要调整大小
        if self.use_residual:
            if x.shape != x_in.shape:
                x_in = F.interpolate(x_in, size=x.shape[2:], mode='bilinear', align_corners=True)
            return x + x_in
        else:
            return x


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi

class AttentionUp(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
            
        self.attn = AttentionGate(in_channels//2, out_channels, out_channels//2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Apply attention mechanism
        x2 = self.attn(x1, x2)
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# Replace the Up modules in your UNet with AttentionUp modules

if __name__ == "__main__":
    # Create dummy input (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 16, 224, 449)
    
    # Initialize model
    model = UNet(n_channels=16, n_classes=1, bilinear=False, attention=True, pretrain=True)
    
    # Check model
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Forward pass
    try:
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print("Model test successful!")
    except Exception as e:
        print(f"Model error: {e}")