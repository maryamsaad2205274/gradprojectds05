from torch.utils.data import DataLoader

train_ds = HeatmapLandmarkDataset(train_pairs, IMG_SIZE, HEATMAP_SIZE, SIGMA)
val_ds   = HeatmapLandmarkDataset(val_pairs, IMG_SIZE, HEATMAP_SIZE, SIGMA)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
