from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model = HRNetKeypoint(17, HEATMAP_SIZE).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

@torch.no_grad()
def val_loss():
    model.eval()
    total = 0.0
    for x, hm, _pts in val_loader:
        x = x.to(device)
        hm = hm.to(device)
        pred = model(x)
        loss = criterion(pred, hm)
        total += loss.item() * x.size(0)
    return total / len(val_ds)

def train_one_epoch():
    model.train()
    total = 0.0
    for x, hm, _pts in tqdm(train_loader, desc="Train", leave=False):
        x = x.to(device)
        hm = hm.to(device)

        pred = model(x)
        loss = criterion(pred, hm)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item() * x.size(0)
    return total / len(train_ds)

best = 1e9
CKPT_PATH = "/content/drive/MyDrive/projectdataset/best_hrnet_17kp.pth"

for epoch in range(1, 21):
    tr = train_one_epoch()
    vl = val_loss()
    print(f"Epoch {epoch:02d} | train_loss={tr:.6f} | val_loss={vl:.6f}")

    if vl < best:
        best = vl
        torch.save({"model": model.state_dict()}, CKPT_PATH)
        print("✅ Saved best checkpoint:", CKPT_PATH)
