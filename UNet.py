import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== 模組1：基礎卷積區塊 ==========
class DoubleConv(nn.Module):
    """將兩組卷積層和 ReLU 激活函數組合在一起"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    """定義資料的前向傳遞流程"""
    def forward(self, x):
        return self.double_conv(x)

# ========== 模組2：UNet 主體 ==========
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        # Encoder：收縮路徑(UNet-下採樣)
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128) #輸入通道數64，輸出通道數128。64*2=128
        self.pool2 = nn.MaxPool2d(2)

        # 中間層
        self.middle = DoubleConv(128, 256)

        # Decoder：擴展路徑(UNet-上採樣)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # 通道數從256到128；特徵圖的寬度、高度放大2倍
        self.upconv2 = DoubleConv(256, 128)  
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv1 = DoubleConv(128, 64)   

        # 輸出層
        self.output_layer = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.down1(x)  # 第一層下採樣後的特徵圖
        x2 = self.down2(self.pool1(x1))  # 第二層下採樣後的特徵圖

        # Middle
        x3 = self.middle(self.pool2(x2))

        # Decoder
        x = self.up2(x3)
        x = self.upconv2(torch.cat([x, x2], dim=1))
        x = self.up1(x)
        x = self.upconv1(torch.cat([x, x1], dim=1))

        # Output
        return self.output_layer(x)

# ========== 模組3：實際使用 ==========
if __name__ == "__main__":
    # 模擬一張 128x128 的灰階影像
    input_tensor = torch.randn(1, 1, 128, 128)  
    target_mask = torch.randint(0, 2, (1, 1, 128, 128)).float()  

    # 建立模型
    model = UNet(in_channels=1, out_channels=1)
    output = model(input_tensor)

    print("輸出尺寸：", output.shape)

    # 計算損失（Binary segmentation）
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(output, target_mask)
    print("損失值：", loss.item())
