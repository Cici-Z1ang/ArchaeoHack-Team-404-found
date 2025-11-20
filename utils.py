print(">>> LOADED utils.py WITH HOG + KNN <<<")

import numpy as np
from PIL import Image
import json
import os
import skimage
import sklearn

from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier


# ============================================================
# 1. Preprocess: 自动裁剪 + 居中 + 缩放到 128×128
# ============================================================

def preprocess(img):
    img = img.convert("L")
    arr = np.array(img)

    # 寻找非白色像素区域
    coords = np.column_stack(np.where(arr < 250))

    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        cropped = arr[y_min:y_max+1, x_min:x_max+1]
    else:
        cropped = arr

    # 转回 PIL 缩放
    cropped_img = Image.fromarray(cropped)
    cropped_img.thumbnail((128, 128))

    # 放进 128×128 中心图
    final = Image.new("L", (128, 128), 255)
    w, h = cropped_img.size
    final.paste(cropped_img, ((128 - w)//2, (128 - h)//2))

    return np.array(final)


# ============================================================
# 2. HOG 特征提取（关键）
# ============================================================

def extract_hog_features(img_arr):
    """
    输入：128×128 的 numpy 灰度图
    输出：HOG 特征向量 (稳定，高维但很稳)
    """
    features = hog(
        img_arr,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    return features


# ============================================================
# 3. 加载训练数据
# ============================================================

def load_training_data(data_dir=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if data_dir is None:
        data_dir = os.path.join(base_dir, "data")

    X = []
    y = []

    if not os.path.exists(data_dir):
        return np.array(X), np.array(y)

    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            continue

        for file in os.listdir(class_dir):
            if not file.endswith(".png"):
                continue
            path = os.path.join(class_dir, file)

            img = Image.open(path)
            arr = preprocess(img)
            feat = extract_hog_features(arr)

            X.append(feat)
            y.append(label)

    return np.array(X), np.array(y)


# ============================================================
# 4. 训练 KNN 模型（自动，按需）
# ============================================================

_knn_model = None

def get_knn_model():
    global _knn_model

    if _knn_model is not None:
        return _knn_model

    X, y = load_training_data()
    if len(X) == 0:
        return None

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    _knn_model = knn

    print(f">>> Trained KNN on {len(X)} samples")
    return _knn_model


# ============================================================
# 5. 找最近标签（HOG + KNN）
# ============================================================

def find_nearest_label(img_arr, data_dir="data"):
    model = get_knn_model()
    if model is None:
        return None

    feat = extract_hog_features(img_arr.reshape(128, 128))
    pred = model.predict([feat])[0]
    return pred


# ==============================================================
# 6. JSON glyph lookup — 自动适配 dict 或 list
# ==============================================================

def get_glyph_info(pred_id, json_path=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if json_path is None:
        json_path = os.path.join(base_dir, "glyph_data.json")

    if not os.path.exists(json_path):
        print("^ glyph_data.json missing!")
        return {}

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # case 1: JSON 是 dict
    if isinstance(data, dict):
        return data.get(pred_id, {})

    # case 2: JSON 是 list
    if isinstance(data, list):
        new_dict = {}
        for item in data:
            # 注意这里！！！！！
            if "gardiner_num" in item:
                new_dict[item["gardiner_num"]] = item
        return new_dict.get(pred_id, {})

    return {}





