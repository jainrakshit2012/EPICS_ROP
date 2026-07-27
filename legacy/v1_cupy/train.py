import os
os.environ["CUPY_ACCELERATORS"] = ""

import cupy as cp
import random
import matplotlib.pyplot as plt

from models import CNN
from data_loader import load_dataset
from utils import softmax, cross_entropy, accuracy

# 🔥 SETTINGS
batch_size = 1
lr = 0.001
epochs = 5

# 🔥 LOAD DATA
X, y = load_dataset("C:/Users/vishw/Downloads/EPICS/HVDROPDB_RetCam_Neo_Classification")

print("Dataset size:", len(X))

# 🔥 TRAIN / VAL SPLIT
split = int(0.8 * len(X))
X_train = X[:split]
y_train = y[:split]

X_val = X[split:]
y_val = y[split:]

# 🔥 DEBUG CLASS DISTRIBUTION
print("Train ROP:", sum(y_train))
print("Train Normal:", len(y_train) - sum(y_train))

print("Val ROP:", sum(y_val))
print("Val Normal:", len(y_val) - sum(y_val))

# 🔥 MODEL
model = CNN()

# 🔥 TRAINING LOOP
for epoch in range(epochs):

    # Shuffle training data
    combined = list(zip(X_train, y_train))
    random.shuffle(combined)
    X_train, y_train = zip(*combined)

    total_loss = 0
    total_acc = 0
    count = 0

    for i in range(0, len(X_train), batch_size):

        batch_X = cp.stack(X_train[i:i+batch_size]).astype(cp.float32)
        batch_y = cp.array(y_train[i:i+batch_size])

        # Forward
        out, features = model.forward(batch_X)
        probs = softmax(out)

        # Loss + Accuracy
        loss = cross_entropy(probs, batch_y)
        acc = accuracy(probs, batch_y)

        total_loss += loss
        total_acc += acc
        count += 1

        # Backprop (FC layer only)
        m = len(batch_y)
        dout = probs
        dout[cp.arange(m), batch_y] -= 1
        dout /= m

        grad_W = features.T @ dout
        grad_b = dout.mean(axis=0) * m

        model.W4 -= lr * grad_W
        model.b4 -= lr * grad_b

    print(f"\nEpoch {epoch}, Loss: {total_loss/count}, Acc: {total_acc/count}")

    # 🔥 VALIDATION (FULL DATASET — FIXED)
    if len(X_val) > 0:

        val_total_acc = 0
        val_count = 0

        for i in range(0, len(X_val), batch_size):

            val_X = cp.stack(X_val[i:i+batch_size]).astype(cp.float32)
            val_y = cp.array(y_val[i:i+batch_size])

            val_out, _ = model.forward(val_X)
            val_probs = softmax(val_out)

            val_acc = accuracy(val_probs, val_y)

            val_total_acc += val_acc
            val_count += 1

        print("Validation Acc:", val_total_acc / val_count)

# 🔥 SAVE MODEL
cp.save("W4.npy", model.W4)
cp.save("b4.npy", model.b4)

print("\nModel saved ✅")


# 🔥 VISUALIZATION FUNCTION
def visualize_predictions(model, X, y, num_samples=5):

    for i in range(min(num_samples, len(X))):

        img = X[i]
        label = y[i]

        batch = cp.stack([img]).astype(cp.float32)

        out, _ = model.forward(batch)
        probs = cp.asnumpy(out)

        pred = probs.argmax(axis=1)[0]

        pred_label = "ROP" if pred == 1 else "Normal"
        true_label = "ROP" if label == 1 else "Normal"

        img_cpu = cp.asnumpy(img).transpose(1,2,0)

        plt.imshow(img_cpu)
        plt.title(f"Pred: {pred_label} | True: {true_label}")
        plt.axis("off")
        plt.show()


# 🔥 CALL VISUALIZATION (FIXED POSITION)
visualize_predictions(model, X_val, y_val, num_samples=5)