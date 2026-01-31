!pip -q install timm
import timm
import torch.nn as nn
import torch

class HRNetKeypoint(nn.Module):
    def __init__(self, num_keypoints=17, heatmap_size=64):
        super().__init__()
        self.backbone = timm.create_model(
            "hrnet_w18", pretrained=True, features_only=True, out_indices=(3,)
        )
        ch = self.backbone.feature_info.channels()[0]

        self.head = nn.Sequential(
            nn.Conv2d(ch, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_keypoints, 1)
        )
        self.heatmap_size = heatmap_size

    def forward(self, x):
        feat = self.backbone(x)[0]
        hm = self.head(feat)
        hm = nn.functional.interpolate(hm, size=(self.heatmap_size, self.heatmap_size),
                                       mode="bilinear", align_corners=False)
        return hm
