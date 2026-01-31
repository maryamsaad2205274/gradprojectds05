@torch.no_grad()
def heatmaps_to_points(hm):  # hm: (1,K,H,W)
    B, K, H, W = hm.shape
    flat = hm.view(B, K, -1)
    idx = torch.argmax(flat, dim=-1)
    ys = (idx // W).float()
    xs = (idx % W).float()
    return torch.stack([xs, ys], dim=-1)  # (1,K,2)
