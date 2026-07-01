# -*- coding: utf-8 -*-
"""
基于SOM分区的TrAdaBoost迁移学习 - 黄土高原分区滑坡易发性制图
利用全区因子栅格和分区栅格，对每个分区训练TrAdaBoost-SVM模型并预测易发性
"""

import os
import numpy as np
import rasterio
import geopandas as gpd
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve
import joblib
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
from shapely.geometry import Point
import warnings

warnings.filterwarnings('ignore')
import matplotlib

matplotlib.use('Agg')  # 避免 GUI 线程冲突
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

# ===============================
# 0. 用户配置路径（请根据实际情况修改）
# ===============================

# --- 全区因子栅格文件夹（与分区栅格完全对齐）---
factor_dir = r"D:\arcgis_practice\Loess_Plateau\Data"  # 存放 elev.tif, slope.tif, ...
factor_names = ['DEM_1km_WGS84', 'Slope', 'Aspect', 'plan_curv', 'profile_curv', 'Curvatu','RAIN_1km_WGS84' ,
               'annual_mean_T' , 'FVC_1km_WGS84']
# --- 分区栅格 ---
zone_raster_path = r"D:\SOM_Data\Data_Result_1km\SOM_result4\SOM_5.tif"  # 像元值1~5

# --- 山西源域数据 ---
shanxi_landslide_path = r"D:/GISData/loess_zaihai/shanxi_inrange.shp"
shanxi_nonlandslide_path = r"D:\arcgis_practice\no_landsliderange\non-supervisory\shanxi_nolandslide.shp"
scaler_path = r"D:\arcgis_practice\Loess_Plateau\SVM2\scaler.pkl"

# --- 黄土高原全区的滑坡和非滑坡点（用于每个分区的目标域样本）---
# 如果已经有每个分区独立的点文件，可以直接使用；否则提供全区点文件，脚本会自动按分区拆分
loess_landslide_path = r"D:\arcgis_practice\Loess_Plateau\Data\landslide_zone_250.shp"  # 全区滑坡点
loess_nonlandslide_path = r"D:\arcgis_practice\Loess_Plateau\Data\non_landslide_zone250.shp"  # 全区非滑坡点


# --- 输出根目录 ---
output_root = r"D:\arcgis_practice\Loess_Plateau\TrAdaBoost_Results_BySvM0701"
os.makedirs(output_root, exist_ok=True)

# --- TrAdaBoost 超参数 ---
N_ITER = 50  # 一般 30~100


# ===============================
# 1. 辅助函数
# ===============================

def load_raster_stack(factor_dir, factor_names, ref_raster_path=None):
    """
    从文件夹中读取所有因子栅格，返回堆栈 (rows, cols, n_factors) 和 profile
    如果提供 ref_raster_path，则以其为参考进行重采样（假设所有因子已经对齐）
    """
    # 使用第一个因子获取形状和profile
    first_path = os.path.join(factor_dir, f"{factor_names[0]}.tif")
    with rasterio.open(first_path) as src:
        rows, cols = src.shape
        profile = src.profile
        dtype = src.dtypes[0]
        transform = src.transform
        crs = src.crs

    stack = np.zeros((rows, cols, len(factor_names)), dtype=np.float32)
    for i, name in enumerate(factor_names):
        tif_path = os.path.join(factor_dir, f"{name}.tif")
        with rasterio.open(tif_path) as src:
            stack[:, :, i] = src.read(1).astype(np.float32)
    return stack, profile


def extract_points_in_zone(point_gdf, zone_map, zone_id, transform):
    """
    从点GeoDataFrame中提取落在指定分区内的点，并返回因子值（需要结合因子堆栈）
    这里先只返回点的坐标和几何，后续再提取因子值
    """
    points_in_zone = []
    for idx, row in point_gdf.iterrows():
        geom = row.geometry
        x, y = geom.x, geom.y
        # 根据坐标获取分区ID
        col, row_idx = ~transform * (x, y)
        col, row_idx = int(col), int(row_idx)
        if 0 <= row_idx < zone_map.shape[0] and 0 <= col < zone_map.shape[1]:
            if zone_map[row_idx, col] == zone_id:
                points_in_zone.append((x, y))
    return points_in_zone


def extract_values_from_points(points, factor_stack, transform):
    """
    根据点坐标列表，从因子堆栈中提取每个点的因子值
    返回 (n_points, n_factors) 数组
    """
    values = []
    rows, cols, n_factors = factor_stack.shape
    for (x, y) in points:
        col, row = ~transform * (x, y)
        row, col = int(row), int(col)
        if 0 <= row < rows and 0 <= col < cols:
            vals = factor_stack[row, col, :]
            if not np.any(np.isnan(vals)):
                values.append(vals)
    if len(values) == 0:
        return np.array([])
    return np.array(values)


# ===============================
# 2. TrAdaBoost-RF 类（与之前相同）
# ===============================

class TrAdaBoostSVM:
    def __init__(self, base_svm_params=None, n_estimators=20):

        self.n_estimators = n_estimators

        self.base_svm_params = base_svm_params if base_svm_params else {
            'kernel': 'rbf',
            'C': 10,
            'gamma': 'scale',
            'probability': True,
            'class_weight': 'balanced',
            'random_state': 42
        }

        self.models = []
        self.betas = []

    def fit(self, X_source, y_source, X_target, y_target):
        n_source = X_source.shape[0]
        n_target = X_target.shape[0]

        # 初始化权重
        weights = np.concatenate([np.ones(n_source) / n_source, np.ones(n_target) / n_target])
        X_all = np.vstack([X_source, X_target])
        y_all = np.hstack([y_source, y_target])

        beta0 = 1 / (1 + np.sqrt(2 * np.log(n_source) / self.n_estimators))

        for t in range(self.n_estimators):

            svm = SVC(**self.base_svm_params)

            svm.fit(X_all, y_all, sample_weight=weights)

            self.models.append(svm)

            y_pred_source = svm.predict(X_source)
            y_pred_target = svm.predict(X_target)

            # 计算目标域误差
            error_rate = np.sum(weights[n_source:] * (y_pred_target != y_target)) / np.sum(weights[n_source:])
            error_rate = min(max(error_rate, 1e-15), 0.5)  # 避免除零和过大
            beta = error_rate / (1 - error_rate)
            self.betas.append(beta)

            # 更新权重
            weights[n_source:] = weights[n_source:] * (beta ** (-(y_pred_target != y_target).astype(float)))
            weights[:n_source] = weights[:n_source] * (beta0 ** ((y_pred_source != y_source).astype(float)))
            weights /= np.sum(weights)

    def predict_proba(self, X):
        """
        使用 log(1/beta) 加权各轮 RF 的预测概率（经典 TrAdaBoost）
        """
        proba_sum = np.zeros(X.shape[0])
        total_weight = np.sum([np.log(1 / b) for b in self.betas])
        for beta, model in zip(self.betas, self.models):
            proba_sum += np.log(1 / beta) * model.predict_proba(X)[:, 1]
        return proba_sum / total_weight

    def predict(self, X, thresh=0.5):
        return (self.predict_proba(X) >= thresh).astype(int)


# ===============================
# 3. 主程序
# ===============================

def main():
    print("=" * 60)
    print("基于SOM分区的TrAdaBoost迁移学习 - 分区易发性制图")
    print("=" * 60)

    # ---- 3.1 加载全区因子栅格堆栈和分区栅格 ----
    print("\n[1] 加载全区因子栅格和分区栅格...")
    factor_stack, factor_profile = load_raster_stack(factor_dir, factor_names)
    rows, cols, n_factors = factor_stack.shape
    print(f"   因子栅格尺寸: {rows} x {cols}, 因子数: {n_factors}")

    with rasterio.open(zone_raster_path) as src:
        zone_map = src.read(1).astype(np.int32)
        zone_transform = src.transform
        # 确保分区栅格与因子栅格对齐（理论上应该一致）
        if zone_map.shape != (rows, cols):
            raise ValueError("分区栅格与因子栅格尺寸不一致，请先重采样对齐")
    unique_zones = np.unique(zone_map)
    unique_zones = unique_zones[~np.isnan(unique_zones)]
    print(f"   分区ID: {unique_zones}")

    # ---- 3.2 加载山西源域数据 ----
    # ---- 3.2 加载山西源域数据（直接使用黄土高原因子栅格） ----
    print("\n[2] 加载山西源域数据（从黄土高原因子栅格提取）...")

    def extract_from_points_gdf(point_gdf, factor_stack, transform):
        """从点GeoDataFrame提取因子值，返回特征数组"""
        values = []
        for geom in point_gdf.geometry:
            x, y = geom.x, geom.y
            col, row = ~transform * (x, y)
            row, col = int(row), int(col)
            if 0 <= row < factor_stack.shape[0] and 0 <= col < factor_stack.shape[1]:
                vals = factor_stack[row, col, :]
                if not np.any(np.isnan(vals)):
                    values.append(vals)
        return np.array(values)

    # 读取山西点数据
    gdf_source_ls = gpd.read_file(shanxi_landslide_path)
    gdf_source_nls = gpd.read_file(shanxi_nonlandslide_path)

    # 使用黄土高原因子堆栈和转换参数（factor_stack, zone_transform）
    X_source_ls = extract_from_points_gdf(gdf_source_ls, factor_stack, zone_transform)
    X_source_nls = extract_from_points_gdf(gdf_source_nls, factor_stack, zone_transform)

    print(f"   山西滑坡点有效: {len(X_source_ls)}, 非滑坡点: {len(X_source_nls)}")

    if len(X_source_ls) == 0 or len(X_source_nls) == 0:
        raise ValueError("山西点未能从黄土高原因子栅格提取到有效值，请检查坐标系统是否一致。")

    # 平衡源域
    min_source = min(len(X_source_ls), len(X_source_nls))
    np.random.seed(42)
    idx_ls = np.random.choice(len(X_source_ls), min_source, replace=False)
    idx_nls = np.random.choice(len(X_source_nls), min_source, replace=False)
    X_source = np.vstack([X_source_ls[idx_ls], X_source_nls[idx_nls]])
    y_source = np.hstack([np.ones(min_source), np.zeros(min_source)])

    # 标准化（使用已有的scaler）
    scaler = joblib.load(scaler_path)
    X_source_scaled = scaler.transform(X_source)
    print(f"   源域最终样本数: {len(X_source_scaled)} (滑坡:{min_source}, 非滑坡:{min_source})")

    # ---- 3.3 加载黄土高原全区滑坡和非滑坡点（用于各分区目标域） ----
    print("\n[3] 加载黄土高原全区滑坡/非滑坡点...")
    gdf_landslides = gpd.read_file(loess_landslide_path)
    gdf_nonlandslides = gpd.read_file(loess_nonlandslide_path)
    print(f"   全区滑坡点: {len(gdf_landslides)}, 非滑坡点: {len(gdf_nonlandslides)}")

    # =========================
    # 所有分区 ROC 总图
    # =========================
    plt.figure(figsize=(8, 6))
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    roc_data_list = []

    # ---- 3.4 对每个分区进行迁移学习 ----
    for zone_id in unique_zones:
        print(f"\n[4] 处理分区 {zone_id}")

        # 4.1 提取该分区内的滑坡点和非滑坡点
        # 通过点坐标和分区栅格判断
        def points_in_zone(gdf, zone_id, zone_map, transform):
            points = []
            for geom in gdf.geometry:
                x, y = geom.x, geom.y
                col, row = ~transform * (x, y)
                row, col = int(row), int(col)
                if 0 <= row < zone_map.shape[0] and 0 <= col < zone_map.shape[1]:
                    if zone_map[row, col] == zone_id:
                        points.append((x, y))
            return points

        landslide_pts = points_in_zone(gdf_landslides, zone_id, zone_map, zone_transform)
        nonlandslide_pts = points_in_zone(gdf_nonlandslides, zone_id, zone_map, zone_transform)
        print(f"   分区内滑坡点: {len(landslide_pts)}, 非滑坡点: {len(nonlandslide_pts)}")

        if len(landslide_pts) == 0 or len(nonlandslide_pts) == 0:
            print(f"   分区内样本不足，跳过")
            continue

        # 提取因子值
        X_target_ls = extract_values_from_points(landslide_pts, factor_stack, zone_transform)
        X_target_nls = extract_values_from_points(nonlandslide_pts, factor_stack, zone_transform)
        print(f"   有效滑坡点值: {len(X_target_ls)}, 有效非滑坡点值: {len(X_target_nls)}")
        if len(X_target_ls) == 0 or len(X_target_nls) == 0:
            continue

        # 平衡目标域
        min_target = min(len(X_target_ls), len(X_target_nls))
        np.random.seed(42)
        idx_ls_t = np.random.choice(len(X_target_ls), min_target, replace=False)
        idx_nls_t = np.random.choice(len(X_target_nls), min_target, replace=False)
        X_target = np.vstack([X_target_ls[idx_ls_t], X_target_nls[idx_nls_t]])
        y_target = np.hstack([np.ones(min_target), np.zeros(min_target)])
        X_target_scaled = scaler.transform(X_target)
        print(f"   目标域训练样本数: {len(X_target_scaled)} (滑坡:{min_target}, 非滑坡:{min_target})")

        # 4.2 划分测试集（如果样本数足够）
        if len(X_target_scaled) >= 20:
            from sklearn.model_selection import train_test_split
            X_tr, X_te, y_tr, y_te = train_test_split(X_target_scaled, y_target,
                                                      test_size=0.3, random_state=42, stratify=y_target)
            print(f"   划分训练: {len(X_tr)}, 测试: {len(X_te)}")
        else:
            X_tr, y_tr = X_target_scaled, y_target
            X_te, y_te = None, None
            print(f"   样本较少，全部用于训练")

        # 4.3 训练 TrAdaBoost-RF
        svm_params = {
            'kernel': 'rbf',
            'C': 10,
            'gamma': 'scale',
            'probability': True,
            'class_weight': 'balanced',
            'random_state': 42
        }

        tr_model = TrAdaBoostSVM(
            base_svm_params=svm_params,
            n_estimators=N_ITER
        )
        tr_model.fit(X_source_scaled, y_source, X_tr, y_tr)

        # 4.4 评估（如果有测试集）
        # 新增：保存每个分区的 ROC 曲线信息


        # 在原有分区循环内替换 ROC 绘图部分：
        if X_te is not None:
            proba_te = tr_model.predict_proba(X_te)
            auc_val = roc_auc_score(y_te, proba_te)
            print(f"   测试集 AUC: {auc_val:.4f}")

            fpr, tpr, _ = roc_curve(y_te, proba_te)
            roc_data_list.append({'zone_id': zone_id, 'fpr': fpr, 'tpr': tpr, 'auc': auc_val})

            # ROC 曲线
            fpr, tpr, _ = roc_curve(y_te, proba_te)
            auc_val = roc_auc_score(y_te, proba_te)

            # 添加到总 ROC 图
            plt.plot(
                fpr,
                tpr,
                color=colors[int(zone_id) % len(colors)],
                linewidth=2,
                label=f'Zone {zone_id} (AUC={auc_val:.3f})'
            )

            # =========================
            # 混淆矩阵
            # =========================

            # 概率转类别
            y_pred_class = tr_model.predict(X_te, thresh=0.5)

            # 计算混淆矩阵
            cm = confusion_matrix(y_te, y_pred_class)

            print("\n混淆矩阵:")
            print(cm)

            # TN FP
            # FN TP
            tn, fp, fn, tp = cm.ravel()

            # 各项指标
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp + 1e-15)
            recall = tp / (tp + fn + 1e-15)
            specificity = tn / (tn + fp + 1e-15)
            f1 = 2 * precision * recall / (precision + recall + 1e-15)

            print(f"Accuracy   : {accuracy:.4f}")
            print(f"Precision  : {precision:.4f}")
            print(f"Recall(TPR): {recall:.4f}")
            print(f"Specificity: {specificity:.4f}")
            print(f"F1-score   : {f1:.4f}")

            # 分类报告
            print("\nClassification Report:")
            print(classification_report(y_te, y_pred_class))

            # =========================
            # 绘制混淆矩阵图
            # =========================

            fig, ax = plt.subplots(figsize=(6, 6))

            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=["Non-Landslide", "Landslide"]
            )

            disp.plot(
                cmap='Blues',
                ax=ax,
                colorbar=False
            )

            plt.title(f'Zone {zone_id} - Confusion Matrix')

            cm_path = os.path.join(
                output_root,
                f'Zone_{zone_id}_ConfusionMatrix.png'
            )

            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"   混淆矩阵图保存至: {cm_path}")
            # ---- 统一绘制所有分区 ROC ----
            plt.figure(figsize=(8, 6))
            colors = ['r', 'g', 'b', 'm', 'c', 'y', 'k']  # 不同分区颜色
            for i, roc_data in enumerate(roc_data_list):
                plt.plot(roc_data['fpr'], roc_data['tpr'], color=colors[i % len(colors)],
                         label=f"Zone {int(roc_data['zone_id'])} - AUC={roc_data['auc']:.3f}")
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves for All Zones')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_root, 'All_Zones_ROC.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   所有分区 ROC 图保存至: {os.path.join(output_root, 'All_Zones_ROC.png')}")

        # 4.5 预测该分区所有像元的易发性
        print(f"   预测分区 {zone_id} 易发性...")
        # 获取该分区的掩膜
        mask_zone = (zone_map == zone_id)
        # 提取掩膜内所有像元的因子值
        indices = np.where(mask_zone)
        n_pixels = len(indices[0])
        if n_pixels == 0:
            continue
        # 构建预测矩阵
        X_zone = np.zeros((n_pixels, n_factors), dtype=np.float32)
        for i in range(n_factors):
            factor_data = factor_stack[:, :, i]
            X_zone[:, i] = factor_data[indices[0], indices[1]]
        # 处理无效值
        X_zone = np.nan_to_num(X_zone)
        X_zone_scaled = scaler.transform(X_zone)
        proba_zone = tr_model.predict_proba(X_zone_scaled)

        # 创建输出栅格（全为NaN）
        result_raster = np.full((rows, cols), np.nan, dtype=np.float32)
        result_raster[indices[0], indices[1]] = proba_zone

        # 保存GeoTIFF（使用因子栅格的profile）
        out_tif = os.path.join(output_root, f'Zone_{zone_id}_TrAdaBoost_sus.tif')
        profile = factor_profile.copy()
        profile.update(dtype='float32', compress='lzw', nodata=np.nan)
        with rasterio.open(out_tif, 'w', **profile) as dst:
            dst.write(result_raster, 1)
        print(f"   易发性图保存至: {out_tif}")
    # =========================
    # 保存所有分区 ROC 总图
    # =========================

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.title('ROC Curves of All SOM Zones')

    plt.legend(loc='lower right')

    plt.grid(True)

    all_roc_path = os.path.join(output_root, 'All_Zones_ROC.png')

    plt.savefig(
        all_roc_path,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(f"\n五个分区 ROC 总图保存至: {all_roc_path}")

    print("\n所有分区处理完成！")


if __name__ == "__main__":
    main()