# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')   # 🔥 强制使用无GUI后端
import os
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.mask import mask
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, confusion_matrix, auc
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from rasterio.enums import Resampling
from shapely.geometry import Point

# 自定义函数：检查点是否在山西的边界内
def point_within_boundary(point, boundary):
    """检查一个点是否在山西边界内"""
    point_geom = Point(point)  # 将坐标转化为Point对象
    return boundary.contains(point_geom)  # 返回点是否在边界内
# ===============================
# 1. 路径设置
# ===============================
factor_paths = {

    "elev": r"E:\Data\Data\DEM_1km_WGS84.tif",
    "slope": r"E:\Data\Data\slope.tif",
    "aspect": r"E:\Data\Data\aspect.tif",
    "plan_curv": r"E:\Data\Data\plan_curv.tif",
    "profile_curv": r"E:\Data\Data\profile_curv.tif",
    "curvature": r"E:\Data\Data\Curvatu.tif",
    "T": r"E:\Data\Data\annual_mean_T.tif",
    "rain": r"E:\Data\Data\RAIN_1km_WGS84.tif",
    "FVC":r"E:\Data\Data\FVC_1km_WGS84.tif"

}
feature_names = list(factor_paths.keys())

landslide_path = r"E:\Data\Data/landslide_Plateau.shp"
nonlandslide_path = r"E:\Data\Data\non_landslide_Plteau_prj.shp"
shanxi_shp_path =r"E:\Data\LoessPlateauRegion\LoessPlateauRegion.shp"
  # 山西区域边界shp文件路径

output_folder = r"E:\Data\Model\RF_Plateau"
os.makedirs(output_folder, exist_ok=True)

# ===============================
# 2. 读取栅格因子（统一大小和CRS）
# ===============================
print("读取因子...")

def read_resample(path, ref_profile):
    with rasterio.open(path) as src:
        data = src.read(
            1,
            out_shape=(ref_profile['height'], ref_profile['width']),
            resampling=Resampling.bilinear
        )
    return data.astype(np.float32)

# 参考栅格
ref_name, ref_path = list(factor_paths.items())[0]
with rasterio.open(ref_path) as src:
    ref_profile = src.profile
    factor_arrays = [src.read(1).astype(np.float32)]

# 其他因子
for name, path in list(factor_paths.items())[1:]:
    factor_arrays.append(read_resample(path, ref_profile))

factor_stack = np.stack(factor_arrays, axis=-1)
rows, cols, num_factors = factor_stack.shape

# ===============================
# 3. 样本提取
# ===============================
print("提取样本...")

def extract_values(shp_path):
    gdf = gpd.read_file(shp_path).to_crs(ref_profile['crs'])
    coords = [(geom.x, geom.y) for geom in gdf.geometry]
    values = []
    transform = ref_profile['transform']

    for x, y in coords:
        col, row = ~transform * (x, y)
        row, col = int(row), int(col)
        if 0 <= row < rows and 0 <= col < cols:
            values.append(factor_stack[row, col, :])
    return np.array(values)

X_ls = extract_values(landslide_path)
X_nls = extract_values(nonlandslide_path)

# 平衡样本
np.random.seed(42)
X_nls = X_nls[np.random.choice(len(X_nls), len(X_ls), replace=False)]

X = np.vstack((X_ls, X_nls))
y = np.hstack((np.ones(len(X_ls)), np.zeros(len(X_nls))))

# ===============================
# 4. 标准化
# ===============================
print("标准化...")
X = np.nan_to_num(X)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(output_folder, "scaler.pkl"))

# ===============================
# 5. 划分数据（分层！）
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# ===============================
# 6. 随机森林训练
# ===============================
print("训练随机森林...")
'''
第一次加入降水的参数设置
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)'''
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=10,
    min_samples_split=5,
    class_weight='balanced',
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)
'''model.fit(X_train, y_train)
joblib.dump(model, os.path.join(output_folder, "rf_model.pkl"))

'''
model.fit(X_train, y_train)

# 保存模型
joblib.dump(model, os.path.join(output_folder, "rf_model.pkl"))

# 保存训练数据
joblib.dump(
    {
        "X_train": X_train,
        "y_train": y_train,
        "feature_names": feature_names
    },
    os.path.join(output_folder, "rf_training_data.pkl")
)
# ===============================
# 7. 模型评估
# ===============================
print("模型评估...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.savefig(os.path.join(output_folder, "confusion_matrix.png"), dpi=300)
plt.close()

# ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f'AUC={roc_auc_score(y_test, y_prob):.3f}')
plt.plot([0,1],[0,1],'--')
plt.legend()
plt.title("ROC Curve")
plt.savefig(os.path.join(output_folder, "roc_curve.png"), dpi=300)
plt.close()

# ===============================
# 8. 特征重要性
# ===============================
print("计算特征重要性...")

importances = model.feature_importances_

indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "feature_importance.png"), dpi=300)
plt.close()

# ===============================
# 9. 分块预测（后处理：裁剪到山西区域）
# ===============================
# ===============================
# 9. 分块预测（正确裁剪方式）
# ===============================
print("开始预测...")

pred_map = np.zeros((rows, cols), dtype=np.float32)
block_size = 500

# 分块预测
for i in range(0, rows, block_size):
    for j in range(0, cols, block_size):
        i_end = min(i + block_size, rows)
        j_end = min(j + block_size, cols)

        block = factor_stack[i:i_end, j:j_end, :]
        block_reshape = block.reshape(-1, num_factors)
        block_reshape = scaler.transform(np.nan_to_num(block_reshape))

        pred = model.predict_proba(block_reshape)[:, 1]
        pred_map[i:i_end, j:j_end] = pred.reshape(block.shape[0], block.shape[1])

        print(f"完成块: {i}-{i_end}, {j}-{j_end}")

# ===============================
# 9.5 使用shp裁剪（关键修改）
# ===============================
print("裁剪到山西范围...")

# 读取山西边界
shanxi_gdf = gpd.read_file(shanxi_shp_path).to_crs(ref_profile['crs'])

# 保存临时tif（用于mask）
temp_tif = os.path.join(output_folder, "temp_pred.tif")
ref_profile.update(dtype=rasterio.float32, compress='lzw')

with rasterio.open(temp_tif, 'w', **ref_profile) as dst:
    dst.write(pred_map, 1)

# 裁剪
with rasterio.open(temp_tif) as src:
    out_image, out_transform = mask(
        src,
        shanxi_gdf.geometry,
        crop=False,
        nodata=np.nan
    )

pred_map = out_image[0]

# ===============================
# 10. 输出
# ===============================
output_tif = os.path.join(output_folder, "RF_result.tif")

ref_profile.update(
    dtype=rasterio.float32,
    transform=out_transform,
    compress='lzw',
    nodata=np.nan
)

with rasterio.open(output_tif, 'w', **ref_profile) as dst:
    dst.write(pred_map.astype(np.float32), 1)
print("训练样本数量:", X_train.shape)
print("特征数量:", len(feature_names))
print("类别分布:", np.bincount(y_train.astype(int)))
print("完成！结果输出:", output_tif)

