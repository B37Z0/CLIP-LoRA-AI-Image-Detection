"""
Dual-stream CNN baseline (Yousaf et al. 2022 reference)

- Spatial domain ResNet18 stream
- Frequency-domain stream (grayscale -> FFT log-magnitude -> small CNN)
- Fused via concatenation into binary classifier head

Reproduction baseline for project and comparison to CLIP-ViT-based methods.
Frequency stream follows paper preprocessing (YCbCr conversion, per-channel
DFT, single-level Haar DWT) for 18 features. Uses a distinct transform
(resize + tensor) since the frequency branch needs raw pixel values;
normalization is done internally in forward() for the spatial branch only.

Some notable differences due to scoping constraints:
- DFT and DWT are each processed by their own small CNNs instead of 
  stacking them into the 18-feature vector to ResNet50 (avoids having 
  to adapt the first 3-channel conv layer to 18 channels).
- Paper uses probability-level fusion (averaging softmax outputs) on
  the two streams. Feature-level concatenation is used here instead.
- Spatial stream augments (Gaussian blur + JPEG compression) are not
  used. Sorry Wang et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Fixed 2x2 Haar filters for single-level 2D DWT via stride-2 convolution.
# - Add C=1 dim for single-channel input and normalize (1/2).
HAAR_FILTERS = torch.tensor([
    [[1., 1.], [1., 1.]],        # LL - approximation (low-freq content)
    [[1., 1.], [-1., -1.]],      # LH - horizontal detail (row edges)
    [[1., -1.], [1., -1.]],      # HL - vertical detail (column edges)
    [[1., -1.], [-1., 1.]],      # HH - diagonal detail (corners/texture)
]).unsqueeze(1) / 2.0  # (K, C, H, W) -> (4, 1, 2, 2)


def build_cnn_transform(image_size=224):
    """
    Resize and convert to tensor with values in [0, 1].
    """
    return T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])

def rgb_to_ycbcr(x):
    """
    [0,1] RGB -> YCbCr (BT.601).
    """
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return torch.stack([y, cb, cr], dim=1) # (B, 3, H, W)


class FrequencyBranch(nn.Module):
    """
    YCbCr -> per-channel DFT (real+imaginary) & single-level Haar DWT
    (4 sub-bands each). Processed by two separate CNNs and concatenated.
    """

    def __init__(self, out_dim=128):
        super().__init__()
        half_dim = out_dim // 2

        self.dft_conv = nn.Sequential(
            nn.Conv2d(6, 16, 3, stride=2, padding=1), nn.ReLU(), # (3 channels) * (2 = real, imaginary) = 6 channels
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dft_fc = nn.Linear(64, half_dim)
 
        self.dwt_conv = nn.Sequential(
            nn.Conv2d(12, 16, 3, stride=2, padding=1), nn.ReLU(), # (3 channels) * (4 sub-bands) = 12 channels
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dwt_fc = nn.Linear(32, out_dim - half_dim)
 
        self.register_buffer("haar_filters", HAAR_FILTERS)
 
    def _dft_features(self, ycbcr):
        # Real + imaginary components separately. Don't take magnitude 
        # because the phases retain structural information (edges, positions)
        # that magnitude would discard. Artifacts could show up in either,
        spectrum = torch.fft.fftshift(torch.fft.fft2(ycbcr), dim=(-2, -1))
        return torch.cat([spectrum.real, spectrum.imag], dim=1) # (B, 6, H, W)
 
    def _dwt_features(self, ycbcr):
        # Apply 4 Haar filters for 3 channels independently. Can't do this 
        # with groups=3 so the channels are folded into the batch dim such that
        # each channel is processed as a C=1 image. Unfold back to C=12 - this is 
        # possible because the 4 sub-bands per batch*channel are contiguous in memory.
        b, c, h, w = ycbcr.shape
        x = ycbcr.reshape(b * c, 1, h, w)
        bands = F.conv2d(x, self.haar_filters, stride=2) # (B*3, 4, H/2, W/2)
        return bands.reshape(b, c * 4, h // 2, w // 2)   # (B, 12, H/2, W/2)
 
    def forward(self, x):
        ycbcr = rgb_to_ycbcr(x)
 
        dft_feat = self.dft_fc(self.dft_conv(self._dft_features(ycbcr)).flatten(1))
        dwt_feat = self.dwt_fc(self.dwt_conv(self._dwt_features(ycbcr)).flatten(1))
 
        return torch.cat([dft_feat, dwt_feat], dim=1)


class DualStreamCNN(nn.Module):
    def __init__(self, freq_dim=128):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.spatial_backbone = nn.Sequential(*list(resnet.children())[:-1]) # drop fc
        spatial_dim = resnet.fc.in_features # 512 for resnet18

        self.frequency_branch = FrequencyBranch(out_dim=freq_dim)
        self.classifier = nn.Linear(spatial_dim + freq_dim, 2)

        self.register_buffer("imagenet_mean", IMAGENET_MEAN)
        self.register_buffer("imagenet_std", IMAGENET_STD)

    def forward(self, x):
        spatial_input = (x - self.imagenet_mean) / self.imagenet_std
        spatial_features = self.spatial_backbone(spatial_input).flatten(1)

        freq_features = self.frequency_branch(x)

        fused = torch.cat([spatial_features, freq_features], dim=1)
        return self.classifier(fused)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DualStreamCNN().to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

    dummy_input = torch.rand(4, 3, 224, 224, device=device) # rand for [0,1] range, not randn
    logits = model(dummy_input)
    loss = logits.float().pow(2).mean()
    loss.backward()
    print(f"Forward + backward pass successful. logits.shape={logits.shape}")

"""
Trainable params: 11,214,466 / 11,214,466 (100.0%)
Forward + backward pass successful. logits.shape=torch.Size([4, 2])
"""