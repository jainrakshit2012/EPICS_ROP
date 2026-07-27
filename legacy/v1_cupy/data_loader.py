import os
import cv2
import cupy as cp

def load_dataset(folder):
    X, y = [], []

    for subfolder in os.listdir(folder):
        subpath = os.path.join(folder, subfolder)

        # skip non-folders
        if not os.path.isdir(subpath):
            continue

        print("Reading folder:", subfolder)

        for file in os.listdir(subpath):
            img_path = os.path.join(subpath, file)

            # skip non-images
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img = cv2.imread(img_path)

            if img is None:
                print("Failed to read:", img_path)
                continue

            img = cv2.resize(img, (2048, 2048))
            img = img.transpose(2,0,1) / 255.0

            img = cp.array(img, dtype=cp.float32)

            X.append(img)

            # binary labels
            if "ROP" in subfolder:
                y.append(1)
            else:
                y.append(0)

    print("Total images loaded:", len(X))

    return X, y