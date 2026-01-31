IMG_SIZE = 256  # must match training
HEATMAP_SIZE = 64

def preprocess_image(img_path):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(img_path)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h0, w0 = img_rgb.shape[:2]

    # resize
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    # normalize (ImageNet)
    img = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img - mean) / std

    x = torch.from_numpy(img_norm).permute(2,0,1).unsqueeze(0)  # (1,3,H,W)
    return x, img_resized, (w0, h0)
