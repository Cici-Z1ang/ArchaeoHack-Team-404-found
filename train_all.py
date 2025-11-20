import os
import numpy as np
from PIL import Image
from utils import preprocess, extract_hog_features
from sklearn.neighbors import KNeighborsClassifier
import joblib
from tqdm import tqdm

DATA_ROOT = "data"   # 你的 data 文件夹（有 A1, A2, W7, X8 等）

X = []
y = []

print("🔍 Scanning all folders...")

for label in sorted(os.listdir(DATA_ROOT)):
    class_dir = os.path.join(DATA_ROOT, label)
    if not os.path.isdir(class_dir):
        continue

    print(f"\n📁 Processing class: {label}")

    for file in tqdm(os.listdir(class_dir)):
        if file.lower().endswith(".png"):
            img_path = os.path.join(class_dir, file)

            try:
                img = Image.open(img_path)
                arr = preprocess(img)
                feat = extract_hog_features(arr)

                X.append(feat)
                y.append(label)

            except Exception as e:
                print(f"⚠️ Error processing {img_path}: {e}")

X = np.array(X)
y = np.array(y)

print(f"\n✅ Loaded {len(X)} samples in total.")
print(f"🧩 Number of classes: {len(set(y))}")

# =====================================================
# Train KNN
# =====================================================
print("\n🚀 Training KNN (k=3)...")

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

print("🎉 KNN trained successfully!")

# =====================================================
# Save the KNN model + training data (optional)
# =====================================================
joblib.dump(knn, "knn_model.pkl")
np.save("training_X.npy", X)
np.save("training_y.npy", y)

print("\n💾 Saved: knn_model.pkl, training_X.npy, training_y.npy")
