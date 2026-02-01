from PIL import Image, ImageDraw

def draw_points(image_path, points, out_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    w, h = img.size
    r = max(3, int(min(w, h) * 0.006))  # radius scales with image

    for p in points:
        if isinstance(p, dict):
            x, y = float(p["x"]), float(p["y"])
        else:
            x, y = float(p[0]), float(p[1])

        draw.ellipse((x-r, y-r, x+r, y+r), fill="red", outline="red")

    img.save(out_path)
