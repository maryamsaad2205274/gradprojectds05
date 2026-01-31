@torch.no_grad()
def predict_first_5_unlabeled():
    img_paths = sorted(glob.glob(os.path.join(UNLABELED_DIR, "*.*")))[:5]
    print("Using images:")
    for p in img_paths:
        print(" -", os.path.basename(p))

    scale = IMG_SIZE / HEATMAP_SIZE  # 256 / 64 = 4

    for img_path in img_paths:
        x, img_resized, _ = preprocess_image(img_path)
        x = x.to(device)

        pred_hm = model(x).cpu()                  # (1,17,64,64)
        pred_pts_hm = heatmaps_to_points(pred_hm)[0].numpy()
        pred_pts_img = pred_pts_hm * scale        # back to 256x256 coords

        # draw landmarks
        out_img = img_resized.copy()
        for (px, py) in pred_pts_img:
            cv2.circle(out_img, (int(px), int(py)), 3, (255, 0, 0), -1)

        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(OUT_DIR, f"{base}_pred.png")

        cv2.imwrite(out_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
        print("Saved:", out_path)

    print("✅ Done: predictions for first 5 unlabeled images")

predict_first_5_unlabeled()
