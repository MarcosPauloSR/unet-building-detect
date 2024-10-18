import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_transform=None, mask_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.image_names = os.listdir(images_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.images_dir, image_name)
        mask_path = os.path.join(self.masks_dir, image_name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            mask = (mask > 0).float()  # Binarização

        return image, mask

dataset = SegmentationDataset(images_dir='train/images_old/',
                              masks_dir='train/masks_old/',
                              image_transform=image_transform,
                              mask_transform=mask_transform)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.double_conv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = self.double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = self.double_conv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = self.double_conv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottom = self.double_conv(512, 1024)

        # Decoder
        self.upconv4 = self.up_conv(1024, 512)
        self.dec4 = self.double_conv(1024, 512)

        self.upconv3 = self.up_conv(512, 256)
        self.dec3 = self.double_conv(512, 256)

        self.upconv2 = self.up_conv(256, 128)
        self.dec2 = self.double_conv(256, 128)

        self.upconv1 = self.up_conv(128, 64)
        self.dec1 = self.double_conv(128, 64)

        # Camada final
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def double_conv(self, in_channels, out_channels):
        """(conv => BN => ReLU) * 2"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  
            nn.ReLU(inplace=True),
        )

    def up_conv(self, in_channels, out_channels):
        """Upscaling"""
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)      # [N, 64, H, W]
        x1p = self.pool1(x1)   # [N, 64, H/2, W/2]

        x2 = self.enc2(x1p)    # [N, 128, H/2, W/2]
        x2p = self.pool2(x2)   # [N, 128, H/4, W/4]

        x3 = self.enc3(x2p)    # [N, 256, H/4, W/4]
        x3p = self.pool3(x3)   # [N, 256, H/8, W/8]

        x4 = self.enc4(x3p)    # [N, 512, H/8, W/8]
        x4p = self.pool4(x4)   # [N, 512, H/16, W/16]

        # Bottleneck
        x5 = self.bottom(x4p)  # [N, 1024, H/16, W/16]

        # Decoder
        x6 = self.upconv4(x5)  # [N, 512, H/8, W/8]
        x6 = torch.cat([x6, x4], dim=1)  # [N, 1024, H/8, W/8]
        x6 = self.dec4(x6)     # [N, 512, H/8, W/8]

        x7 = self.upconv3(x6)  # [N, 256, H/4, W/4]
        x7 = torch.cat([x7, x3], dim=1)  # [N, 512, H/4, W/4]
        x7 = self.dec3(x7)     # [N, 256, H/4, W/4]

        x8 = self.upconv2(x7)  # [N, 128, H/2, W/2]
        x8 = torch.cat([x8, x2], dim=1)  # [N, 256, H/2, W/2]
        x8 = self.dec2(x8)     # [N, 128, H/2, W/2]

        x9 = self.upconv1(x8)  # [N, 64, H, W]
        x9 = torch.cat([x9, x1], dim=1)  # [N, 128, H, W]
        x9 = self.dec1(x9)     # [N, 64, H, W]

        # Saída
        output = self.final_conv(x9)  # [N, out_channels, H, W]
        return output
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet(in_channels=3, out_channels=1)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()

images, masks = next(iter(dataloader))
images = images.to(device)
masks = masks.to(device)

with torch.no_grad():
    outputs = model(images)
    probs = torch.sigmoid(outputs)
    preds = (probs > 0.5).float()


# Instanciação do modelo
model = model.to(device)

# Otimizador e função de perda
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

__all__ = ['dataloader', 'model', 'criterion', 'optimizer', 'device']
