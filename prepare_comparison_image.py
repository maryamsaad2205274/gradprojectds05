IMG_SIZE = 256
HEATMAP_SIZE = 64

# load original image
img_bgr = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h0, w0 = img_rgb.shape[:2]

# resize for model
img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

# scale GT landmarks to resized image
sx = IMG_SIZE / w0
sy = IMG_SIZE / h0
gt_resized = gt_pts.copy()
gt_resized[:,0] *= sx
gt_resized[:,1] *= sy
