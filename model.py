import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import timm
from torchvision import transforms

import matplotlib.pyplot as plt

# metrics imports
from scipy.optimize import linear_sum_assignment
from skimage import segmentation
from numba import jit


def _label_overlap(x, y):
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint32)
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1
    return overlap


def _intersection_over_union(masks_true, masks_pred):
    overlap = _label_overlap(masks_true, masks_pred)
    n_pred = np.sum(overlap, axis=0, keepdims=True)
    n_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pred + n_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou


def dice(gt, seg):
    if np.count_nonzero(gt) == 0 and np.count_nonzero(seg) == 0:
        return 1.0
    if np.count_nonzero(gt) == 0 and np.count_nonzero(seg) > 0:
        return 0.0
    union = np.count_nonzero(np.logical_and(gt, seg))
    inter = np.count_nonzero(gt) + np.count_nonzero(seg)
    return 2 * union / inter


def _true_positive(iou, th):
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= th).astype(float) - iou / (2 * n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= th
    return match_ok.sum()


def eval_tp_fp_fn(masks_true, masks_pred, threshold=0.5):
    num_true = masks_true.max()
    num_pred = masks_pred.max()
    if num_pred > 0:
        iou = _intersection_over_union(masks_true, masks_pred)[1:, 1:]
        tp = _true_positive(iou, threshold)
        fp = int(num_pred - tp)
        fn = int(num_true - tp)
    else:
        tp = fp = fn = 0
    return tp, fp, fn


def remove_boundary_cells(mask):
    W, H = mask.shape
    bd = np.ones((W, H), dtype=bool)
    bd[2:W-2, 2:H-2] = False
    bd_ids = np.unique(mask[bd])
    for i in bd_ids:
        if i != 0:
            mask[mask == i] = 0
    new_label, _, _ = segmentation.relabel_sequential(mask)
    return new_label


def get_efficient_net():
    return torch.hub.load(
        'NVIDIA/DeepLearningExamples:torchhub',
        'nvidia_efficientnet_b0', pretrained=True
    )


class EfficientNetBackbone(nn.Module):
    """Pretrained EfficientNet backbone adapted for grayscale images"""
    def __init__(self, model_name='efficientnet_b0',
                 input_channels=1, freeze=True):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained=True,
            features_only=True, out_indices=(2, 3, 4)
        )
        if input_channels == 1:
            conv1 = self.model.conv_stem
            self.model.conv_stem = nn.Conv2d(
                1, conv1.out_channels,
                kernel_size=conv1.kernel_size,
                stride=conv1.stride,
                padding=conv1.padding,
                bias=(conv1.bias is not None)
            )
            with torch.no_grad():
                self.model.conv_stem.weight[:] = conv1.weight.sum(dim=1, keepdim=True)
        if freeze:
            for p in self.model.parameters(): p.requires_grad = False

    def forward(self, x):
        feats = self.model(x)
        return feats[-1], feats[1], feats[2]


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6,
                      dilation=6, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12,
                      dilation=12, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18,
                      dilation=18, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels*5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]
        feats = [b(x) for b in (self.branch1, self.branch2,
                                 self.branch3, self.branch4)]
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=size,
                           mode='bilinear', align_corners=False)
        feats.append(gp)
        return self.project(torch.cat(feats, dim=1))


class DeepLabV3PlusDecoder(nn.Module):
    """DeepLabV3+ decoder head for segmentation"""
    def __init__(self, num_classes, low_level_channels,
                 mid_level_channels, aspp_in_channels):
        super().__init__()
        self.aspp = ASPP(aspp_in_channels, 256)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1), nn.ReLU(inplace=True)
        )
        self.mid_level_conv = nn.Sequential(
            nn.Conv2d(mid_level_channels, 32, 1), nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Conv2d(256+48, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256+32, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.classifier = nn.Conv2d(128, num_classes, 1)

    def forward(self, x, low, mid):
        x = self.aspp(x)
        x = F.interpolate(x, size=low.shape[2:],
                          mode='bilinear', align_corners=False)
        low = self.low_level_conv(low)
        x = F.relu(self.conv1(torch.cat([x, low], dim=1)))
        x = F.relu(self.conv2(x))
        x = F.interpolate(x, size=mid.shape[2:],
                          mode='bilinear', align_corners=False)
        mid = self.mid_level_conv(mid)
        x = F.relu(self.conv3(torch.cat([x, mid], dim=1)))
        x = F.relu(self.conv4(x))
        x = F.interpolate(x, scale_factor=4,
                          mode='bilinear', align_corners=False)
        return self.classifier(x)


class SegmentationModel(nn.Module):
    def __init__(self, backbone_name='efficientnet_b0',
                 num_classes=3, input_channels=1):
        super().__init__()
        self.backbone = EfficientNetBackbone(
            backbone_name, input_channels=input_channels, freeze=True)
        dummy = torch.zeros(1, input_channels, 512, 512)
        with torch.no_grad(): out, low, mid = self.backbone(dummy)
        self.decoder = DeepLabV3PlusDecoder(
            num_classes=num_classes,
            low_level_channels=low.shape[1],
            mid_level_channels=mid.shape[1],
            aspp_in_channels=out.shape[1]
        )

    def forward(self, x): return self.decoder(*self.backbone(x))
    def freeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = False
    def unfreeze_backbone(self, n_layers=10):
        layers = list(self.backbone.model.children())
        for layer in layers[-n_layers:]:
            for p in layer.parameters(): p.requires_grad = True


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir,
                 image_ext='.tif', mask_ext='.tif', transform=None):
        self.image_dir, self.mask_dir = image_dir, mask_dir
        self.image_ext, self.mask_ext = image_ext, mask_ext
        self.transform = transform
        self.images = sorted([f for f in os.listdir(image_dir)
                              if f.endswith(image_ext)])
    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        fname = self.images[idx]
        img = Image.open(os.path.join(self.image_dir, fname)).convert('L')
        base = os.path.splitext(fname)[0]

        # attempt to find mask file with tif or tiff extension
        mask_name = f"{base}_label{self.mask_ext}"
        mask_path = os.path.join(self.mask_dir, mask_name)
        if not os.path.exists(mask_path):
            # try alternative extension
            alt_ext = '.tiff' if self.mask_ext.lower() == '.tif' else '.tif'
            alt_name = f"{base}_label{alt_ext}"
            alt_path = os.path.join(self.mask_dir, alt_name)
            if os.path.exists(alt_path):
                mask_path = alt_path
            else:
                raise FileNotFoundError(f"Mask file for {fname} not found: tried {mask_path} and {alt_path}")

        mask = Image.open(mask_path)
        if self.transform: img = self.transform(img)
        m = transforms.functional.resize(
            mask, (512,512), interpolation=transforms.InterpolationMode.NEAREST)
        m = torch.from_numpy(np.array(m)).long()
        return img, m


def save_predictions(model, loader, device, save_dir='predictions'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for bi, (imgs, _) in enumerate(loader):
            imgs = imgs.to(device)
            out = model(imgs)
            out = F.interpolate(out, size=(512,512),
                                mode='bilinear', align_corners=False)
            preds = out.argmax(1).cpu().numpy()
            for i, p in enumerate(preds):
                Image.fromarray(p.astype(np.uint8), 'L').save(
                    os.path.join(save_dir, f'pred_{bi}_{i}.png')
                )
    model.train()


if __name__ == '__main__':
    # configs
    image_dir, mask_dir = 'resized/images', 'resized/labels'
    bs, epochs, lr = 4, 15, 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.Resize((512,512)), transforms.ToTensor()])

    # data
    full = SegmentationDataset(image_dir, mask_dir, transform=transform)
    t_sz = int(0.8*len(full)); v_sz = len(full)-t_sz
    train_ds, val_ds = random_split(full, [t_sz, v_sz])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    # model
    model = SegmentationModel('efficientnet_b0',3,1).to(device)
    model.unfreeze_backbone(n_layers=len(list(model.backbone.model.children())))
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    # logs
    train_losses, val_losses = [], []
    val_prec, val_rec, val_f1 = [], [], []
    wd = 'model_weights'; os.makedirs(wd, exist_ok=True)

    for e in range(epochs):
        model.train(); run_l=0
        for imgs,msk in train_loader:
            imgs,msk = imgs.to(device),msk.to(device)
            out = model(imgs)
            out = F.interpolate(out, size=msk.shape[1:], mode='bilinear', align_corners=False)
            loss = crit(out,msk)
            opt.zero_grad(); loss.backward(); opt.step()
            run_l += loss.item()*imgs.size(0)
        tr_l = run_l/len(train_loader.dataset); train_losses.append(tr_l)

        # validation
        model.eval(); rv=0
        # metrics accum
        tp=fp=fn=0; tot_true=tot_pred=0
        with torch.no_grad():
            for imgs,msk in val_loader:
                imgs,msk = imgs.to(device),msk.to(device)
                out = model(imgs)
                out = F.interpolate(out, size=msk.shape[1:], mode='bilinear', align_corners=False)
                loss = crit(out,msk); rv += loss.item()*imgs.size(0)

                preds = out.argmax(1).cpu().numpy()
                true_masks = msk.cpu().numpy()
                for gt,pred in zip(true_masks, preds):
                    gt = remove_boundary_cells(gt)
                    pred = remove_boundary_cells(pred)
                    gt,_,_   = segmentation.relabel_sequential(gt)
                    pred,_,_ = segmentation.relabel_sequential(pred)
                    ct_true = gt.max(); ct_pred = pred.max()
                    tp_i,fp_i,fn_i = eval_tp_fp_fn(gt,pred)
                    tp+=tp_i; fp+=fp_i; fn+=fn_i
                    tot_true+=ct_true; tot_pred+=ct_pred
        va_l = rv/len(val_loader.dataset); val_losses.append(va_l)
        prec = tp/ tot_pred if tot_pred>0 else 0
        rec  = tp/ tot_true if tot_true>0 else 0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        val_prec.append(prec); val_rec.append(rec); val_f1.append(f1)

        print(f"Epoch {e+1}/{epochs} — Train Loss: {tr_l:.4f} — Val Loss: {va_l:.4f} "
              f"— Prec: {prec:.4f} — Rec: {rec:.4f} — F1: {f1:.4f}")

    # final save
    save_predictions(model, val_loader, device, save_dir='val_preds')
    torch.save(model.state_dict(), os.path.join(wd,'weights_final.pth'))
    print("Saved val preds and final weights.")

    # plot
    plt.figure()
    epochs_range = range(1,epochs+1)
    plt.plot(epochs_range, train_losses, marker='o', label='Train Loss')
    plt.plot(epochs_range, val_losses,   marker='o', label='Val Loss')
    plt.plot(epochs_range, val_f1,       marker='o', label='Val F1')
    plt.title('Metrics over Epochs')
    plt.xlabel('Epoch'); plt.ylabel('Value')
    plt.grid(True); plt.legend()
    plt.savefig('loss_plot.png')
    print("Saved metric plot to loss_plot.png")
