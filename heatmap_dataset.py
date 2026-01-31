import numpy as np

HEATMAP_SIZE = 64   # 256 -> 64 (downsample x4)
SIGMA = 2.0         # gaussian spread

def gaussian_2d(shape, center, sigma):
    H, W = shape
    x0, y0 = center
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    g = np.exp(-((xx - x0)**2 + (yy - y0)**2) / (2 * sigma**2))
    return g

def points_to_heatmaps(points_xy, img_size, heatmap_size, sigma=2.0):
    """
    points_xy: (K,2) in resized image coords [0..img_size)
    returns: (K, heatmap_size, heatmap_size)
    """
    K = points_xy.shape[0]
    hms = np.zeros((K, heatmap_size, heatmap_size), dtype=np.float32)

    scale = heatmap_size / img_size
    for k in range(K):
        x, y = points_xy[k]
        hx = x * scale
        hy = y * scale
        hms[k] = gaussian_2d((heatmap_size, heatmap_size), (hx, hy), sigma)
    return hms




import torch
from torch.utils.data import Dataset
import cv2, json, os
import numpy as np

NUM_KP = 17
IMG_SIZE = 256

class HeatmapLandmarkDataset(Dataset):
    def __init__(self, pairs, img_size=256, heatmap_size=64, sigma=2.0):
        self.pairs = pairs
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, json_path = self.pairs[idx]

        # image
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h0, w0 = img_rgb.shape[:2]

        # landmarks
        img_name, (jw, jh), pts = load_landmarks_json(json_path)  # pts in original coords (17,2)

        # resize image
        img_resized = cv2.resize(img_rgb, (self.img_size, self.img_size))

        # scale points
        sx = self.img_size / w0
        sy = self.img_size / h0
        pts_resized = pts.copy()
        pts_resized[:, 0] *= sx
        pts_resized[:, 1] *= sy

        # heatmaps
        hms = points_to_heatmaps(pts_resized, self.img_size, self.heatmap_size, self.sigma)  # (K,Hh,Wh)

        # normalize image (ImageNet)
        img = img_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        img_t = torch.from_numpy(img).permute(2,0,1)        # (3,256,256)
        hm_t  = torch.from_numpy(hms)                       # (17,64,64)
        pts_t = torch.from_numpy(pts_resized).float()       # (17,2) for evaluation

        return img_t, hm_t, pts_t
