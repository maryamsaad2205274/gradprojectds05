import os, glob

print("Labeled images:", len(glob.glob(LABELED_DIR + "/*")))
print("Unlabeled images:", len(glob.glob(UNLABELED_DIR + "/*")))
print("Landmark files:", len(glob.glob(LANDMARK_DIR + "/*")))

print("\nSample labeled files:", glob.glob(LABELED_DIR + "/*")[:5])
print("Sample landmark files:", glob.glob(LANDMARK_DIR + "/*")[:5])

