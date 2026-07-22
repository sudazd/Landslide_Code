# -*- coding: utf-8 -*-
"""基于随机森林的黄土高原滑坡易发性预测。"""

import os
import sys
import types

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import seaborn as sns
import shapefile
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.warp import reproject
from shapely.geometry import shape as shapely_shape

# 某些受 Windows 应用程序控制策略管理的电脑会阻止 scikit-learn 中的
# _gradient_boosting.pyd。当前脚本只使用随机森林，因此先为完全不使用的
# HistGradientBoosting 注册占位模块，避免 sklearn.ensemble 顺带加载该 DLL。
_hist_module_name = "sklearn.ensemble._hist_gradient_boosting.gradient_boosting"
if _hist_module_name not in sys.modules:
    _hist_module = types.ModuleType(_hist_module_name)
    _hist_module.HistGradientBoostingClassifier = None
    _hist_module.HistGradientBoostingRegressor = None
    sys.modules[_hist_module_name] = _hist_module

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import MinMaxScaler


# 1. 路径和参数设置
factor_paths = {
    "elev": r"E:\Data\Data\DEM_1km_WGS84.tif",
    "slope": r"E:\Data\Data\slope.tif",
    "aspect": r"E:\Data\Data\aspect.tif",
    "plan_curv": r"E:\Data\Data\plan_curv.tif",
    "profile_curv": r"E:\Data\Data\profile_curv.tif",
    "curvature": r"E:\Data\Data\Curvatu.tif",
    "T": r"E:\Data\Data\annual_mean_T.tif",
    "rain": r"E:\Data\Data\RAIN_1km_WGS84.tif",
    "FVC": r"E:\Data\Data\FVC_1km_WGS84.tif",
}
feature_names = list(factor_paths)

train_landslide_path = r"E:\Data\landslide_7030_Plateau\landslide_Plateau_70.shp"
test_landslide_path = r"E:\Data\landslide_7030_Plateau\landslide_Plateau_30.shp"
nonlandslide_path = r"E:\Data\Data\non_landslide_Plteau_prj.shp"
region_shp_path = r"E:\Data\LoessPlateauRegion\LoessPlateauRegion.shp"
output_folder = r"E:\Data\Model\RF_Plateau722_3070"

RANDOM_STATE = 42
BLOCK_SIZE = 500


def require_files(paths):
    """在耗时计算开始前检查所有输入文件。"""
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError("以下输入文件不存在：\n" + "\n".join(missing))


def read_vector_file(shp_path):
    """读取矢量文件；GDAL 被系统策略拦截时，使用纯 Python 的 pyshp 后备。"""
    try:
        return gpd.read_file(shp_path)
    except ImportError as exc:
        message = str(exc)
        if "pyogrio" not in message and "fiona" not in message and "GDAL" not in message:
            raise

        prj_path = os.path.splitext(shp_path)[0] + ".prj"
        if not os.path.isfile(prj_path):
            raise ValueError(
                f"GDAL 读取引擎不可用，且找不到坐标系文件：{prj_path}"
            ) from exc

        with open(prj_path, "r", encoding="utf-8-sig") as prj_file:
            crs = CRS.from_wkt(prj_file.read())

        with shapefile.Reader(shp_path) as reader:
            geometries = [
                shapely_shape(item.__geo_interface__)
                for item in reader.iterShapes()
            ]

        print(f"GDAL 读取引擎被系统策略拦截，已使用 pyshp 读取：{os.path.basename(shp_path)}")
        return gpd.GeoDataFrame(geometry=geometries, crs=crs)


require_files(
    [
        *factor_paths.values(),
        train_landslide_path,
        test_landslide_path,
        nonlandslide_path,
        region_shp_path,
    ]
)
os.makedirs(output_folder, exist_ok=True)


# 2. 读取并对齐栅格因子
print("读取并对齐栅格因子……")
ref_name, ref_path = next(iter(factor_paths.items()))

with rasterio.open(ref_path) as src:
    ref_profile = src.profile.copy()
    ref_crs = src.crs
    ref_transform = src.transform
    rows, cols = src.height, src.width
    ref_array = src.read(1, masked=True).filled(np.nan).astype(np.float32)

if ref_crs is None:
    raise ValueError(f"参考栅格没有定义坐标系：{ref_path}")

factor_arrays = [ref_array]


def read_and_align_raster(path):
    """将因子栅格重投影到参考栅格的坐标系、分辨率和网格。"""
    destination = np.full((rows, cols), np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"栅格没有定义坐标系：{path}")
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    return destination


for name, path in list(factor_paths.items())[1:]:
    print(f"对齐因子：{name}")
    factor_arrays.append(read_and_align_raster(path))

factor_stack = np.stack(factor_arrays, axis=-1)
num_factors = factor_stack.shape[-1]


# 3. 提取点位置的因子值
print("提取训练和测试样本……")


def extract_values(shp_path):
    """提取点数据对应的全部栅格因子值，并剔除无效样本。"""
    gdf = read_vector_file(shp_path)
    if gdf.empty:
        raise ValueError(f"点文件为空：{shp_path}")
    if gdf.crs is None:
        raise ValueError(f"点文件没有定义坐标系：{shp_path}")

    gdf = gdf.to_crs(ref_crs)
    values = []
    outside_count = 0
    invalid_count = 0

    for geom in gdf.geometry:
        if geom is None or geom.is_empty or geom.geom_type != "Point":
            invalid_count += 1
            continue
        col_float, row_float = ~ref_transform * (geom.x, geom.y)
        row, col = int(np.floor(row_float)), int(np.floor(col_float))
        if not (0 <= row < rows and 0 <= col < cols):
            outside_count += 1
            continue
        sample = factor_stack[row, col, :]
        if not np.all(np.isfinite(sample)):
            invalid_count += 1
            continue
        values.append(sample)

    if not values:
        raise ValueError(f"没有从点文件中提取到有效样本：{shp_path}")

    result = np.asarray(values, dtype=np.float32)
    print(
        f"{os.path.basename(shp_path)}：有效 {len(result)} 个，"
        f"范围外 {outside_count} 个，无效 {invalid_count} 个"
    )
    return result


X_train_ls = extract_values(train_landslide_path)
X_test_ls = extract_values(test_landslide_path)
X_nls_all = extract_values(nonlandslide_path)


# 4. 划分非滑坡点并组合数据
n_available_nls = len(X_nls_all)
n_total_ls = len(X_train_ls) + len(X_test_ls)
rng = np.random.default_rng(RANDOM_STATE)

# 有效非滑坡点略少时，不进行有放回抽样（会产生重复样本），而是按原训练/测试
# 比例下采样滑坡点，从而保持两个数据集内部的正负样本数量相等。
if n_available_nls < n_total_ls:
    target_train_ls = int(round(n_available_nls * len(X_train_ls) / n_total_ls))
    target_train_ls = min(max(target_train_ls, 1), len(X_train_ls))
    target_test_ls = n_available_nls - target_train_ls
    if target_test_ls < 1 or target_test_ls > len(X_test_ls):
        raise ValueError("有效样本过少，无法同时构建训练集和测试集。")

    removed_train = len(X_train_ls) - target_train_ls
    removed_test = len(X_test_ls) - target_test_ls
    X_train_ls = X_train_ls[rng.permutation(len(X_train_ls))[:target_train_ls]]
    X_test_ls = X_test_ls[rng.permutation(len(X_test_ls))[:target_test_ls]]
    print(
        "有效非滑坡点少于滑坡点，已按原比例下采样滑坡点："
        f"训练集移除 {removed_train} 个，测试集移除 {removed_test} 个。"
    )

n_train_nls = len(X_train_ls)
n_test_nls = len(X_test_ls)
n_required = n_train_nls + n_test_nls
if len(X_nls_all) < n_required:
    raise ValueError(
        "有效非滑坡样本数量不足："
        f"需要 {n_required} 个（训练 {n_train_nls} + 测试 {n_test_nls}），"
        f"实际只有 {len(X_nls_all)} 个。"
    )

shuffled_indices = rng.permutation(len(X_nls_all))
X_train_nls = X_nls_all[shuffled_indices[:n_train_nls]]
X_test_nls = X_nls_all[shuffled_indices[n_train_nls:n_required]]

X_train = np.vstack((X_train_ls, X_train_nls))
y_train = np.hstack((np.ones(len(X_train_ls), dtype=np.int32), np.zeros(len(X_train_nls), dtype=np.int32)))
X_test = np.vstack((X_test_ls, X_test_nls))
y_test = np.hstack((np.ones(len(X_test_ls), dtype=np.int32), np.zeros(len(X_test_nls), dtype=np.int32)))

train_order = rng.permutation(len(X_train))
test_order = rng.permutation(len(X_test))
X_train, y_train = X_train[train_order], y_train[train_order]
X_test, y_test = X_test[test_order], y_test[test_order]

print("训练集形状：", X_train.shape)
print("训练集类别分布 [非滑坡, 滑坡]：", np.bincount(y_train))
print("测试集形状：", X_test.shape)
print("测试集类别分布 [非滑坡, 滑坡]：", np.bincount(y_test))


# 5. 归一化（仅使用训练集拟合，避免测试集信息泄漏）
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(output_folder, "scaler.pkl"))


# 6. 随机森林训练
print("训练随机森林……")
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=10,
    min_samples_split=5,
    class_weight="balanced",
    max_features="sqrt",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
model.fit(X_train, y_train)
joblib.dump(model, os.path.join(output_folder, "rf_model.pkl"))
joblib.dump(
    {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names,
        "random_state": RANDOM_STATE,
    },
    os.path.join(output_folder, "rf_training_and_test_data.pkl"),
)


# 7. 独立测试集评估
print("使用独立测试集评估模型……")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_prob)
print("测试集 AUC：", test_auc)
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-landslide", "Landslide"], yticklabels=["Non-landslide", "Landslide"])
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {test_auc:.3f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "roc_curve.png"), dpi=300, bbox_inches="tight")
plt.close()


# 8. 特征重要性
print("计算特征重要性……")
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=np.asarray(feature_names)[indices], orient="h")
plt.xlabel("Importance")
plt.ylabel("Factor")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "feature_importance.png"), dpi=300, bbox_inches="tight")
plt.close()


# 9. 分块预测
print("开始分块预测……")
pred_map = np.full((rows, cols), np.nan, dtype=np.float32)
for i in range(0, rows, BLOCK_SIZE):
    for j in range(0, cols, BLOCK_SIZE):
        i_end, j_end = min(i + BLOCK_SIZE, rows), min(j + BLOCK_SIZE, cols)
        block = factor_stack[i:i_end, j:j_end, :]
        block_2d = block.reshape(-1, num_factors)
        valid_mask = np.all(np.isfinite(block_2d), axis=1)
        block_prediction = np.full(len(block_2d), np.nan, dtype=np.float32)
        if np.any(valid_mask):
            valid_features = scaler.transform(block_2d[valid_mask])
            block_prediction[valid_mask] = model.predict_proba(valid_features)[:, 1]
        pred_map[i:i_end, j:j_end] = block_prediction.reshape(i_end - i, j_end - j)
        print(f"完成块：{i}-{i_end}, {j}-{j_end}")


# 10. 使用区域 SHP 掩膜，并输出预测栅格
print("将预测结果裁剪到研究区范围……")
region_gdf = read_vector_file(region_shp_path)
if region_gdf.empty:
    raise ValueError(f"区域边界文件为空：{region_shp_path}")
if region_gdf.crs is None:
    raise ValueError(f"区域边界文件没有定义坐标系：{region_shp_path}")
region_gdf = region_gdf.to_crs(ref_crs)

temp_tif = os.path.join(output_folder, "temp_pred.tif")
temp_profile = ref_profile.copy()
temp_profile.update(count=1, dtype=rasterio.float32, compress="lzw", nodata=np.nan)

try:
    with rasterio.open(temp_tif, "w", **temp_profile) as dst:
        dst.write(pred_map, 1)
    with rasterio.open(temp_tif) as src:
        out_image, out_transform = mask(src, region_gdf.geometry, crop=False, nodata=np.nan, filled=True)
finally:
    if os.path.exists(temp_tif):
        os.remove(temp_tif)

pred_map = out_image[0]
output_tif = os.path.join(output_folder, "RF_result.tif")
output_profile = ref_profile.copy()
output_profile.update(
    count=1,
    height=pred_map.shape[0],
    width=pred_map.shape[1],
    dtype=rasterio.float32,
    transform=out_transform,
    compress="lzw",
    nodata=np.nan,
)
with rasterio.open(output_tif, "w", **output_profile) as dst:
    dst.write(pred_map.astype(np.float32), 1)

print("训练样本数量：", len(X_train))
print("测试样本数量：", len(X_test))
print("特征数量：", len(feature_names))
print("训练集类别分布 [非滑坡, 滑坡]：", np.bincount(y_train))
print("测试集类别分布 [非滑坡, 滑坡]：", np.bincount(y_test))
print("测试集 AUC：", test_auc)
print("完成！结果输出：", output_tif)
