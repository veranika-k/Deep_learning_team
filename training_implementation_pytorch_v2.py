import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np


def get_efficient_net():
    return torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)

class EfficientNetBackbone(nn.Module):
    """Pretrained EfficientNet backbone adapted for grayscale images"""
    def __init__(self, model_name='efficientnet_b0', input_channels=1, freeze=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, features_only=True, out_indices=(2, 3, 4))
        if input_channels == 1:
            # Replace first conv to accept 1 channel
            conv1 = self.model.conv_stem
            self.model.conv_stem = nn.Conv2d(1, conv1.out_channels, kernel_size=conv1.kernel_size,
                                             stride=conv1.stride, padding=conv1.padding, bias=conv1.bias is not None)
            # Copy weights for grayscale
            with torch.no_grad():
                self.model.conv_stem.weight[:] = conv1.weight.sum(dim=1, keepdim=True)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        # Returns list of feature maps at different stages
        features = self.model(x)
        # features[-1]: deepest, features[1]: low-level, features[2]: mid-level
        return features[-1], features[1], features[2]

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        b5 = self.global_pool(x)
        b5 = F.interpolate(b5, size=size, mode='bilinear', align_corners=False)
        x = torch.cat([b1, b2, b3, b4, b5], dim=1)
        return self.project(x)

class DeepLabV3PlusDecoder(nn.Module):
    """DeepLabV3+ decoder head for segmentation"""
    def __init__(self, num_classes, low_level_channels, mid_level_channels, aspp_in_channels):
        super().__init__()
        self.aspp = ASPP(aspp_in_channels, 256)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.ReLU(inplace=True)
        )
        self.mid_level_conv = nn.Sequential(
            nn.Conv2d(mid_level_channels, 32, 1),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Conv2d(256 + 48, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256 + 32, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.classifier = nn.Conv2d(128, num_classes, 1)

    def forward(self, x, low_level, mid_level):
        x = self.aspp(x)
        x = F.interpolate(x, size=low_level.shape[2:], mode='bilinear', align_corners=False)
        low_level = self.low_level_conv(low_level)
        x = torch.cat([x, low_level], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.interpolate(x, size=mid_level.shape[2:], mode='bilinear', align_corners=False)
        mid_level = self.mid_level_conv(mid_level)
        x = torch.cat([x, mid_level], dim=1)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return self.classifier(x)

class SegmentationModel(nn.Module):
    """Complete segmentation model with EfficientNet backbone and DeepLabV3+ head"""
    def __init__(self, backbone_name='efficientnet_b0', num_classes=3, input_channels=1):
        super().__init__()
        self.backbone = EfficientNetBackbone(backbone_name, input_channels)
        # Get channel sizes for skip connections
        dummy = torch.zeros(1, input_channels, 512, 512)
        with torch.no_grad():
            out, low, mid = self.backbone(dummy)
        self.decoder = DeepLabV3PlusDecoder(
            num_classes=num_classes,
            low_level_channels=low.shape[1],
            mid_level_channels=mid.shape[1],
            aspp_in_channels=out.shape[1]
        )

    def forward(self, x):
        out, low, mid = self.backbone(x)
        return self.decoder(out, low, mid)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, n_layers=10):
        # Unfreeze last n_layers of backbone
        layers = list(self.backbone.model.children())
        for layer in layers[-n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

# Example usage:
if __name__ == "__main__":
    import torch.optim as optim

    class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_ext='.tif', mask_ext='.tif', transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_ext = image_ext
        self.mask_ext = mask_ext
        self.transform = transform

        self.images = sorted([
            fname for fname in os.listdir(image_dir)
            if fname.endswith(image_ext)
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(self.image_ext, self.mask_ext))

        # Load grayscale image and mask
        image = Image.open(img_path).convert('L')  # Grayscale
        mask = Image.open(mask_path)  # Should contain label classes (0, 1, 2, ...)

        if self.transform:
            image = self.transform(image)

        # Resize and convert mask to tensor
        mask = transforms.functional.resize(mask, (512, 512), interpolation=transforms.InterpolationMode.NEAREST)
        mask = torch.from_numpy(np.array(mask)).long()  # ensure integer labels for loss

        return img, mask

    

    image_dir = 'path/to/images'
    mask_dir = 'path/to/masks'

    train_ds = SegmentationDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegmentationModel(backbone_name='efficientnet_b0', num_classes=3, input_channels=1).to(device)
    # Unfreeze the backbone from the start for fine-tuning
    model.unfreeze_backbone(n_layers=len(list(model.backbone.model.children())))
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Training loop (fine-tuning from the start)
    model.train()
    losses = []
    for epoch in range(15):
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            outputs = torch.nn.functional.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        avg_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
        losses.append(avg_loss)
    

    # Plot and save loss over epochs
    plt.figure()
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('loss_plot.png')
    print("Loss plot saved as 'loss_plot.png'")