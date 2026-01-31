import cv2
import torch
from torch.utils.data import Dataset, DataLoader

IMG_SIZE = 256  # input size for training

class FaceLandmarkDataset(Dataset):
    def __init__(self, pairs, img_size=256):
        self.pairs = pairs
        self.img_size = img_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, json_path = self.pairs[idx]

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h0, w0 = img_rgb.shape[:2]

        img_name, (jw, jh), pts = load_landmarks_json(json_path)

        # Resize image
        img_resized = cv2.resize(img_rgb, (self.img_size, self.img_size))

        # Scale points to resized image
        sx = self.img_size / w0
        sy = self.img_size / h0
        pts_resized = pts.copy()
        pts_resized[:, 0] *= sx
        pts_resized[:, 1] *= sy

        # Normalize image (ImageNet normalization)
        img = img_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        img_t = torch.from_numpy(img).permute(2, 0, 1)       # (3,H,W)
        pts_t = torch.from_numpy(pts_resized).float()        # (17,2)

        return img_t, pts_t, img_path
