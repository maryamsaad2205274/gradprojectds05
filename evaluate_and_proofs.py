@torch.no_grad()
def heatmaps_to_points(hm):  # (B,K,H,W)
    B, K, H, W = hm.shape
    flat = hm.view(B, K, -1)
    idx = torch.argmax(flat, dim=-1)
    ys = (idx // W).float()
    xs = (idx % W).float()
    return torch.stack([xs, ys], dim=-1)

@torch.no_grad()
def evaluate_mre():
    model.eval()
    total_dist = 0.0
    total_pts = 0

    scale = IMG_SIZE / HEATMAP_SIZE  # 256/64 = 4

    for x, _hm, gt_pts in tqdm(val_loader, desc="Eval", leave=False):
        x = x.to(device)
        pred_hm = model(x).cpu()
        pred_pts_hm = heatmaps_to_points(pred_hm)          # heatmap coords
        pred_pts_img = pred_pts_hm * scale                 # image coords

        gt = gt_pts.float()                                # image coords
        pred = pred_pts_img.float()

        dist = torch.sqrt(((pred - gt) ** 2).sum(dim=-1))   # (B,K)
        total_dist += dist.sum().item()
        total_pts += dist.numel()

    return total_dist / total_pts

mre = evaluate_mre()
print(f"✅ Validation Mean Radial Error (pixels): {mre:.3f}")



import matplotlib.pyplot as plt
import numpy as np
import os

PROOF_DIR = "/content/drive/MyDrive/projectdataset/proof_images"
os.makedirs(PROOF_DIR, exist_ok=True)

@torch.no_grad()
def save_proofs(n=5):
    model.eval()
    saved = 0
    scale = IMG_SIZE / HEATMAP_SIZE

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

    for x, _hm, gt_pts in val_loader:
        pred_hm = model(x.to(device)).cpu()
        pred_pts_hm = heatmaps_to_points(pred_hm)
        pred_pts = (pred_pts_hm * scale).numpy()

        x_vis = ((x * std + mean).clamp(0,1)).permute(0,2,3,1).numpy()
        gt = gt_pts.numpy()

        for i in range(x_vis.shape[0]):
            if saved >= n:
                return
            plt.figure(figsize=(5,5))
            plt.imshow(x_vis[i])
            plt.scatter(gt[i][:,0], gt[i][:,1], s=20, marker="o")   # GT
            plt.scatter(pred_pts[i][:,0], pred_pts[i][:,1], s=20, marker="x")  # Pred
            plt.axis("off")
            out = os.path.join(PROOF_DIR, f"proof_{saved}.png")
            plt.savefig(out, bbox_inches="tight", pad_inches=0.1)
            plt.close()
            print("Saved:", out)
            saved += 1

save_proofs(n=5)
