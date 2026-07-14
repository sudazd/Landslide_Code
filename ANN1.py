# -*- coding: utf-8 -*-
import os
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.mask import mask
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# ======================
# 1. 参数设置
# ======================
PATCH = 13
pad = PATCH // 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_SAMPLES = 50000   # 防止爆内存
BATCH_SIZE = 32
EPOCHS = 30

# ======================
# 2. 读取函数
# ======================
def read_and_clip(tif_path, shp):
    with rasterio.open(tif_path) as src:
        out_image, out_transform = mask(src, shp.geometry, crop=True)
        data = out_image[0]

        nodata = src.nodata if src.nodata is not None else -9999
        data[data == nodata] = np.nan

    return data, out_transform, src.crs

# ======================
# 3. 读取数据
# ======================
print("读取山西边界...")
shp = gpd.read_file(r"D:\arcgis_practice\shanxi_boundary.shp")

# ---------- 静态因子 ----------
print("读取静态因子...")
factor_paths = [
    r"D:\arcgis_practice\factorimportance\data\elevation.tif",
    r"D:\arcgis_practice\factorimportance\data\slope.tif",
    r"D:\arcgis_practice\factorimportance\data\aspect.tif",
    # 👉 补齐你的12个因子
]

static_list = []

for i, path in enumerate(factor_paths):
    data, transform, crs = read_and_clip(path, shp)

    if i == 0:
        H, W = data.shape

    data = np.where(np.isnan(data), -9999, data)
    static_list.append(data)

static = np.stack(static_list, axis=-1)
print("Static:", static.shape)

# ---------- 降雨 ----------
print("读取降雨序列...")
rain_files = sorted(glob(r"D:\arcgis_practice\rain_shanxi\raintif\*.tif"))[:48]

rain_list = []

for f in rain_files:
    data, _, _ = read_and_clip(f, shp)

    data[data < 0] = np.nan
    data = np.where(np.isnan(data), -9999, data)

    rain_list.append(data)

rain = np.stack(rain_list, axis=0)
rain = rain[..., np.newaxis]

print("Rain:", rain.shape)

# ---------- 标签 ----------
print("读取标签...")
label_path = r"D:\arcgis_practice\landslide.tif"
label, _, _ = read_and_clip(label_path, shp)
label = np.where(label > 0, 1, 0)

# ======================
# 4. 构建样本
# ======================
def build_samples(static, rain, label):
    samples_s, samples_r, samples_y = [], [], []

    count = 0
    for i in range(pad, H - pad):
        for j in range(pad, W - pad):

            if static[i, j, 0] == -9999:
                continue

            if count > MAX_SAMPLES:
                break

            s_patch = static[i-pad:i+pad+1, j-pad:j+pad+1, :]
            r_patch = rain[:, i-pad:i+pad+1, j-pad:j+pad+1, :]

            samples_s.append(s_patch)
            samples_r.append(r_patch)
            samples_y.append(label[i, j])

            count += 1

    return np.array(samples_s), np.array(samples_r), np.array(samples_y)

print("构建训练样本...")
X_s, X_r, y = build_samples(static, rain, label)

# ======================
# 5. 数据划分
# ======================
X_s_train, X_s_val, X_r_train, X_r_val, y_train, y_val = train_test_split(
    X_s, X_r, y, test_size=0.2, random_state=42
)

train_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_s_train, dtype=torch.float32),
        torch.tensor(X_r_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    ),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_s_val, dtype=torch.float32),
        torch.tensor(X_r_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    ),
    batch_size=BATCH_SIZE
)

# ======================
# 6. 模型定义
# ======================
class RainBranch(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, 64)

    def forward(self, x):
        B, T, H, W, C = x.shape
        x = x.view(B*T, C, H, W)

        x = self.cnn(x)
        x = x.view(B, T, -1)

        x, _ = self.lstm(x)
        x = x[:, -1, :]

        return self.fc(x)

class StaticBranch(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Linear(64, 64)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class TSDNN(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.rain = RainBranch()
        self.static = StaticBranch(in_ch)

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, s, r):
        f1 = self.static(s)
        f2 = self.rain(r)
        x = torch.cat([f1, f2], dim=1)
        return self.fc(x)

model = TSDNN(static.shape[-1]).to(DEVICE)

# ======================
# 7. 训练
# ======================
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print("开始训练...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for s, r, y in train_loader:
        s, r, y = s.to(DEVICE), r.to(DEVICE), y.to(DEVICE)

        pred = model(s, r).squeeze()
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss={total_loss:.4f}")

# ======================
# 8. 预测整图
# ======================
print("开始预测整幅图...")
result = np.full((H, W), -9999, dtype=np.float32)

model.eval()

with torch.no_grad():
    for i in range(pad, H - pad):
        for j in range(pad, W - pad):

            if static[i, j, 0] == -9999:
                continue

            s_patch = static[i-pad:i+pad+1, j-pad:j+pad+1, :]
            r_patch = rain[:, i-pad:i+pad+1, j-pad:j+pad+1, :]

            s_patch = torch.tensor(s_patch).unsqueeze(0).float().to(DEVICE)
            r_patch = torch.tensor(r_patch).unsqueeze(0).float().to(DEVICE)

            pred = model(s_patch, r_patch)
            result[i, j] = pred.item()

# ======================
# 9. 输出tif
# ======================
print("输出结果...")

with rasterio.open(
    r"D:\arcgis_practice\shanxi_lsm.tif",
    "w",
    driver="GTiff",
    height=H,
    width=W,
    count=1,
    dtype="float32",
    crs=crs,
    transform=transform,
    nodata=-9999
) as dst:
    dst.write(result, 1)

print("完成！")