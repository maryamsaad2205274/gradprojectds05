# preprocess for model
img_norm = img_resized.astype(np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
img_norm = ((img_norm - mean) / std).astype(np.float32)

x = torch.from_numpy(img_norm).permute(2,0,1).unsqueeze(0).to(device).float()

model = model.to(device).float()
model.eval()

with torch.no_grad():
    pred_hm = model(x).cpu()
