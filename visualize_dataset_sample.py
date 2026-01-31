import matplotlib.pyplot as plt

train_ds = FaceLandmarkDataset(train_pairs, IMG_SIZE)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2)

imgs, pts, paths = next(iter(train_loader))

# de-normalize for display
mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

img0 = (imgs[0:1]*std + mean).clamp(0,1)[0].permute(1,2,0).numpy()
pts0 = pts[0].numpy()

plt.figure(figsize=(5,5))
plt.imshow(img0)
plt.scatter(pts0[:,0], pts0[:,1], s=20)
plt.title("Resized + scaled landmarks (check correctness)")
plt.axis("off")
plt.show()

print("Example file:", paths[0])
print("Image tensor shape:", imgs.shape)
print("Landmarks tensor shape:", pts.shape)
