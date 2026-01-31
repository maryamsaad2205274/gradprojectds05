import os, glob

print("Labeled images:", len(glob.glob(LABELED_DIR + "/*")))
print("Unlabeled images:", len(glob.glob(UNLABELED_DIR + "/*")))
print("Landmark files:", len(glob.glob(LANDMARK_DIR + "/*")))

print("\nSample labeled files:", glob.glob(LABELED_DIR + "/*")[:5])
print("Sample landmark files:", glob.glob(LANDMARK_DIR + "/*")[:5])

import os, glob

img_paths = sorted(glob.glob(os.path.join(LABELED_DIR, "*.*")))

def find_landmark_file(img_path):
    base = os.path.splitext(os.path.basename(img_path))[0]
    candidates = [
        os.path.join(LANDMARK_DIR, base + ".json"),
        os.path.join(LANDMARK_DIR, base + ".txt"),
        os.path.join(LANDMARK_DIR, base + ".csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

pairs = []
missing = 0
for p in img_paths:
    lm = find_landmark_file(p)
    if lm is None:
        missing += 1
    else:
        pairs.append((p, lm))

print("Total labeled images:", len(img_paths))
print("Pairs found:", len(pairs))
print("Missing landmarks:", missing)

print("\nExample pair:", pairs[0] if pairs else "No pairs found")

import os, glob
sample = sorted(glob.glob(LANDMARK_DIR + "/*"))[0]
print("Sample landmark file:", sample)
print("Extension:", os.path.splitext(sample)[1].lower())

with open(sample, "r", encoding="utf-8", errors="ignore") as f:
    for i in range(10):
        line = f.readline()
        if not line: break
        print(line.strip())

