# -*- coding: utf-8 -*-
"""
使用独立的30%验证点，对已经生成的STRUT-RF易发性TIF进行精度验证。

本程序不训练模型，也不依赖 strut_corrected.py。它只执行：
1. 读取滑坡/非滑坡验证点；
2. 自动去重并剔除与STRUT更新样本坐标重合的验证点；
3. 从STRUT-RF和SOM栅格提取点位概率与分区；
4. 输出总体及Zone1-5的ROC、混淆矩阵和精度指标。

混淆矩阵和分类指标统一使用概率阈值0.5。
"""

from pathlib import Path
import os

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd

WORK = Path(r"E:\Data\Model\RF_Plateau722_3070\STRUT_rf")
MPL_DIR = WORK / ".matplotlib"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib.pyplot as plt
import rasterio
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ==============================
# 1. 路径设置
# ==============================

LANDSLIDE_VALIDATION = Path(
    r"E:\Data\landslide_7030_Plateau\landslide_Plateau_30.shp"
)
NONLANDSLIDE_VALIDATION = Path(
    r"E:\Data\Data\non_landslide_Plteau_prj.shp"
)

# 第2步STRUT更新所用点，仅用于检查并删除验证集重合点。
LANDSLIDE_STRUT_TRAIN = Path(
    r"E:\Data\landslide_7030_Plateau\Landsilide_zone250_train.shp"
)
NONLANDSLIDE_STRUT_TRAIN = Path(
    r"E:\Data\Data\non_landslide_zone250.shp"
)

# 2026-07-22最终预测结果。
MODEL_RASTERS = {
    "STRUT_RF": Path(
        r"E:\Data\Model\RF_Plateau722_3070\STRUT_rf\Prediction2\STRUT_RF_Final.tif"
    ),
}

SOM_RASTER = Path(r"E:\Data\SOFM\SOM_result6\SOM_5.tif")
OUTPUT_DIR = WORK / "AccuracyValidation_30percent_STRUT"

THRESHOLD = 0.5
COORDINATE_DECIMALS = 8
BOOTSTRAP_ITERATIONS = 1000
RANDOM_SEED = 20260722


def check_files():
    paths = [
        LANDSLIDE_VALIDATION,
        NONLANDSLIDE_VALIDATION,
        LANDSLIDE_STRUT_TRAIN,
        NONLANDSLIDE_STRUT_TRAIN,
        SOM_RASTER,
        *MODEL_RASTERS.values(),
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("以下输入文件不存在:\n" + "\n".join(missing))


def coordinate_key(gdf):
    return list(
        zip(
            gdf.geometry.x.round(COORDINATE_DECIMALS),
            gdf.geometry.y.round(COORDINATE_DECIMALS),
        )
    )


def read_points(path, target_crs, label, source):
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(f"样本没有坐标系: {path}")
    gdf = gdf.to_crs(target_crs)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if not np.all(gdf.geometry.geom_type == "Point"):
        raise ValueError(f"样本必须全部为点: {path}")
    gdf["label"] = int(label)
    gdf["sample_source"] = source
    gdf["coord_key"] = coordinate_key(gdf)
    return gdf[["label", "sample_source", "coord_key", "geometry"]]


def build_validation_points(target_crs):
    positive = read_points(
        LANDSLIDE_VALIDATION, target_crs, 1, "landslide_validation_30"
    )
    negative = read_points(
        NONLANDSLIDE_VALIDATION, target_crs, 0, "nonlandslide_validation_30"
    )
    positive_train = read_points(
        LANDSLIDE_STRUT_TRAIN, target_crs, 1, "landslide_strut_train"
    )
    negative_train = read_points(
        NONLANDSLIDE_STRUT_TRAIN, target_crs, 0, "nonlandslide_strut_train"
    )

    positive_original = len(positive)
    negative_original = len(negative)
    positive = positive.drop_duplicates("coord_key").copy()
    negative = negative.drop_duplicates("coord_key").copy()
    positive_duplicate = positive_original - len(positive)
    negative_duplicate = negative_original - len(negative)

    positive_train_keys = set(positive_train["coord_key"])
    negative_train_keys = set(negative_train["coord_key"])
    positive_overlap = positive["coord_key"].isin(positive_train_keys)
    negative_overlap = negative["coord_key"].isin(negative_train_keys)
    positive_overlap_count = int(positive_overlap.sum())
    negative_overlap_count = int(negative_overlap.sum())
    positive = positive[~positive_overlap].copy()
    negative = negative[~negative_overlap].copy()

    conflicts = set(positive["coord_key"]) & set(negative["coord_key"])
    if conflicts:
        raise ValueError(
            f"有{len(conflicts)}个坐标同时被标记为滑坡和非滑坡，请先检查数据"
        )

    points = pd.concat([positive, negative], ignore_index=True)
    points.insert(0, "sample_id", np.arange(len(points), dtype=int))
    points["x"] = points.geometry.x
    points["y"] = points.geometry.y
    summary = {
        "PositiveOriginal": positive_original,
        "NegativeOriginal": negative_original,
        "PositiveDuplicatesRemoved": positive_duplicate,
        "NegativeDuplicatesRemoved": negative_duplicate,
        "PositiveSTRUTOverlapRemoved": positive_overlap_count,
        "NegativeSTRUTOverlapRemoved": negative_overlap_count,
        "PositiveAfterOverlapRemoval": len(positive),
        "NegativeAfterOverlapRemoval": len(negative),
    }
    return points, summary


def raster_signature(path):
    with rasterio.open(path) as src:
        return src.crs, src.shape, src.transform


def verify_raster_alignment():
    reference_path = next(iter(MODEL_RASTERS.values()))
    reference = raster_signature(reference_path)
    for name, path in {**MODEL_RASTERS, "SOM": SOM_RASTER}.items():
        if raster_signature(path) != reference:
            raise ValueError(f"{name}栅格与参考栅格的CRS、尺寸或网格不一致")
    if reference[0] is None:
        raise ValueError("预测栅格没有坐标系")
    return reference[0]


def sample_raster(path, points):
    coordinates = list(zip(points["x"], points["y"]))
    values = np.full(len(points), np.nan, dtype=float)
    with rasterio.open(path) as src:
        bounds = src.bounds
        for i, sample in enumerate(src.sample(coordinates, masked=True)):
            value = sample[0]
            if not np.ma.is_masked(value):
                values[i] = float(value)
        inside = (
            (points["x"].to_numpy() >= bounds.left)
            & (points["x"].to_numpy() < bounds.right)
            & (points["y"].to_numpy() > bounds.bottom)
            & (points["y"].to_numpy() <= bounds.top)
        )
        valid = inside & np.isfinite(values)
        if src.nodata is not None and np.isfinite(src.nodata):
            valid &= values != src.nodata
    return values, valid


def prepare_predictions(points, sample_summary):
    common_valid = np.ones(len(points), dtype=bool)
    sampled = {}
    for model_name, path in MODEL_RASTERS.items():
        probability, valid = sample_raster(path, points)
        sampled[model_name] = probability
        common_valid &= valid

    som_value, som_valid = sample_raster(SOM_RASTER, points)
    rounded_zone = np.rint(som_value)
    som_valid &= np.isin(rounded_zone, [1, 2, 3, 4, 5])
    common_valid &= som_valid

    removed = int((~common_valid).sum())
    points = points.loc[common_valid].copy().reset_index(drop=True)
    points["zone"] = rounded_zone[common_valid].astype(int)
    for model_name, probability in sampled.items():
        probability = probability[common_valid]
        if probability.min() < 0 or probability.max() > 1:
            raise ValueError(
                f"{model_name}概率超出[0,1]: {probability.min()}..{probability.max()}"
            )
        points[f"{model_name}_probability"] = probability

    sample_summary["RasterNoDataOrOutsideRemoved"] = removed
    sample_summary["FinalValidationSamples"] = len(points)
    sample_summary["FinalPositiveSamples"] = int((points["label"] == 1).sum())
    sample_summary["FinalNegativeSamples"] = int((points["label"] == 0).sum())
    return points, sample_summary


def metric_row(model_name, scope, zone, labels, probability):
    prediction = (probability >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    fpr, tpr, _ = roc_curve(labels, probability)
    return {
        "Scope": scope,
        "Zone": zone,
        "Model": model_name,
        "Threshold": THRESHOLD,
        "Samples": len(labels),
        "PositiveSamples": int((labels == 1).sum()),
        "NegativeSamples": int((labels == 0).sum()),
        "ROC_AUC": roc_auc_score(labels, probability),
        "PR_AUC_AP": average_precision_score(labels, probability),
        "KS": np.max(tpr - fpr),
        "Accuracy": accuracy_score(labels, prediction),
        "BalancedAccuracy": balanced_accuracy_score(labels, prediction),
        "Precision_PPV": precision_score(labels, prediction, zero_division=0),
        "Recall_Sensitivity": recall_score(labels, prediction, zero_division=0),
        "Specificity": specificity,
        "NegativePredictiveValue": npv,
        "F1": f1_score(labels, prediction, zero_division=0),
        "MCC": matthews_corrcoef(labels, prediction),
        "CohenKappa": cohen_kappa_score(labels, prediction),
        "BrierScore": brier_score_loss(labels, probability),
        "LogLoss": log_loss(labels, probability, labels=[0, 1]),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def calculate_metrics(points):
    overall_rows = []
    zone_rows = []
    labels = points["label"].to_numpy(dtype=int)
    for model_name in MODEL_RASTERS:
        probability = points[f"{model_name}_probability"].to_numpy(dtype=float)
        overall_rows.append(
            metric_row(model_name, "Overall", "All", labels, probability)
        )
    for zone, group in points.groupby("zone", sort=True):
        zone_labels = group["label"].to_numpy(dtype=int)
        if len(np.unique(zone_labels)) < 2:
            print(f"警告: Zone {zone}只有一个类别，跳过该分区ROC和指标")
            continue
        for model_name in MODEL_RASTERS:
            probability = group[f"{model_name}_probability"].to_numpy(dtype=float)
            zone_rows.append(
                metric_row(model_name, "Zone", int(zone), zone_labels, probability)
            )
    return pd.DataFrame(overall_rows), pd.DataFrame(zone_rows)


def bootstrap_auc(points):
    labels = points["label"].to_numpy(dtype=int)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(RANDOM_SEED)
    boot = {name: np.empty(BOOTSTRAP_ITERATIONS) for name in MODEL_RASTERS}
    probabilities = {
        name: points[f"{name}_probability"].to_numpy(dtype=float)
        for name in MODEL_RASTERS
    }
    for i in range(BOOTSTRAP_ITERATIONS):
        selected = np.r_[
            rng.choice(positive, len(positive), replace=True),
            rng.choice(negative, len(negative), replace=True),
        ]
        for name in MODEL_RASTERS:
            boot[name][i] = roc_auc_score(labels[selected], probabilities[name][selected])
    rows = []
    for name in MODEL_RASTERS:
        rows.append({
            "Model": name,
            "AUC": roc_auc_score(labels, probabilities[name]),
            "CI95_Lower": np.quantile(boot[name], 0.025),
            "CI95_Upper": np.quantile(boot[name], 0.975),
        })
    return pd.DataFrame(rows)


def make_roc_tables(points):
    overall = []
    zones = []
    labels = points["label"].to_numpy(dtype=int)
    for model_name in MODEL_RASTERS:
        probability = points[f"{model_name}_probability"].to_numpy(dtype=float)
        fpr, tpr, thresholds = roc_curve(labels, probability)
        auc_value = roc_auc_score(labels, probability)
        for a, b, c in zip(fpr, tpr, thresholds):
            overall.append({"Model": model_name, "ROC_AUC": auc_value,
                            "FPR": a, "TPR": b, "Threshold": c})
    for zone, group in points.groupby("zone", sort=True):
        zone_labels = group["label"].to_numpy(dtype=int)
        if len(np.unique(zone_labels)) < 2:
            continue
        for model_name in MODEL_RASTERS:
            probability = group[f"{model_name}_probability"].to_numpy(dtype=float)
            fpr, tpr, thresholds = roc_curve(zone_labels, probability)
            auc_value = roc_auc_score(zone_labels, probability)
            for a, b, c in zip(fpr, tpr, thresholds):
                zones.append({"Zone": int(zone), "Model": model_name,
                              "ROC_AUC": auc_value, "FPR": a, "TPR": b,
                              "Threshold": c})
    return pd.DataFrame(overall), pd.DataFrame(zones)


def plot_overall_roc(roc_table):
    figure, axis = plt.subplots(figsize=(7, 6), dpi=180)
    for model_name, group in roc_table.groupby("Model", sort=False):
        auc_value = group["ROC_AUC"].iloc[0]
        axis.plot(group["FPR"], group["TPR"], linewidth=2,
                  label=f"{model_name} (AUC={auc_value:.3f})")
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set(xlabel="False positive rate", ylabel="True positive rate",
             title="ROC curves on independent 30% validation set",
             xlim=(0, 1), ylim=(0, 1))
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "ROC_Overall.png", bbox_inches="tight")
    plt.close(figure)


def plot_zone_roc(zone_roc):
    figure, axis = plt.subplots(figsize=(8, 6), dpi=180)
    colors = plt.cm.tab10(np.linspace(0, 1, 5))
    table = zone_roc[zone_roc["Model"] == "STRUT_RF"]
    for color, (zone, group) in zip(colors, table.groupby("Zone", sort=True)):
        auc_value = group["ROC_AUC"].iloc[0]
        axis.plot(group["FPR"], group["TPR"], color=color, linewidth=2,
                  label=f"Zone {zone} (AUC={auc_value:.3f})")
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set(xlabel="False positive rate", ylabel="True positive rate",
             title="STRUT-RF ROC curves for Zone 1-5",
             xlim=(0, 1), ylim=(0, 1))
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", fontsize=9)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "ROC_All_Zones.png", bbox_inches="tight")
    plt.close(figure)


def draw_matrix(axis, matrix, title):
    axis.imshow(matrix, cmap="Blues", vmin=0)
    totals = matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(matrix, totals, out=np.zeros_like(matrix, dtype=float),
                            where=totals != 0)
    color_threshold = matrix.max() / 2
    for row in range(2):
        for column in range(2):
            axis.text(column, row,
                      f"{matrix[row, column]:,}\n({percentages[row, column]:.1%})",
                      ha="center", va="center",
                      color="white" if matrix[row, column] > color_threshold else "black")
    axis.set(title=title, xlabel="Predicted label", ylabel="True label",
             xticks=[0, 1], yticks=[0, 1],
             xticklabels=["Non-landslide", "Landslide"],
             yticklabels=["Non-landslide", "Landslide"])


def plot_confusion_matrices(points):
    labels = points["label"].to_numpy(dtype=int)
    figure, axis = plt.subplots(figsize=(6, 5), dpi=180)
    probability = points["STRUT_RF_probability"].to_numpy(dtype=float)
    prediction = (probability >= THRESHOLD).astype(int)
    matrix = confusion_matrix(labels, prediction, labels=[0, 1])
    draw_matrix(axis, matrix, f"STRUT_RF overall (threshold={THRESHOLD})")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "Confusion_Matrices_Overall.png", bbox_inches="tight")
    plt.close(figure)

    zones = sorted(points["zone"].unique())
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=180)
    axes = axes.ravel()
    for axis, zone in zip(axes, zones):
        group = points[points["zone"] == zone]
        zone_labels = group["label"].to_numpy(dtype=int)
        probability = group["STRUT_RF_probability"].to_numpy(dtype=float)
        matrix = confusion_matrix(zone_labels, probability >= THRESHOLD, labels=[0, 1])
        draw_matrix(axis, matrix, f"STRUT_RF Zone {zone}")
    for axis in axes[len(zones):]:
        axis.axis("off")
    figure.suptitle(f"STRUT-RF confusion matrices by zone (threshold={THRESHOLD})")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "STRUT_Confusion_Matrices_All_Zones.png",
                   bbox_inches="tight")
    plt.close(figure)


def plot_metric_comparison(overall_metrics, zone_metrics):
    metrics = ["ROC_AUC", "PR_AUC_AP", "Accuracy", "BalancedAccuracy",
               "Precision_PPV", "Recall_Sensitivity", "Specificity", "F1"]
    table = overall_metrics.set_index("Model").loc["STRUT_RF"]
    x = np.arange(len(metrics))
    figure, axis = plt.subplots(figsize=(13, 6), dpi=180)
    axis.bar(x, table[metrics].to_numpy(dtype=float), 0.6, label="STRUT_RF")
    axis.set(title="Overall metrics on independent 30% validation set",
             ylabel="Metric value", xticks=x, xticklabels=metrics, ylim=(0, 1))
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "Metrics_Overall.png", bbox_inches="tight")
    plt.close(figure)

    metrics = ["ROC_AUC", "BalancedAccuracy", "F1", "BrierScore"]
    zones = sorted(zone_metrics["Zone"].unique())
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=180)
    for axis, metric in zip(axes.ravel(), metrics):
        group = zone_metrics.set_index("Zone").loc[zones]
        axis.bar(np.arange(len(zones)), group[metric], 0.6, label="STRUT_RF")
        axis.set(title=metric, xticks=np.arange(len(zones)),
                 xticklabels=[f"Zone {z}" for z in zones])
        if metric != "BrierScore":
            axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.suptitle("Metrics by SOM zone")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "Metrics_All_Zones.png", bbox_inches="tight")
    plt.close(figure)


def main():
    check_files()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_crs = verify_raster_alignment()
    points, sample_summary = build_validation_points(target_crs)
    points, sample_summary = prepare_predictions(points, sample_summary)

    overall_metrics, zone_metrics = calculate_metrics(points)
    auc_confidence = bootstrap_auc(points)
    overall_roc, zone_roc = make_roc_tables(points)

    point_columns = ["sample_id", "label", "sample_source", "x", "y", "zone",
                     "STRUT_RF_probability"]
    points[point_columns].to_csv(OUTPUT_DIR / "Validation_Point_Predictions.csv",
                                 index=False, encoding="utf-8-sig")
    pd.DataFrame([sample_summary]).to_csv(OUTPUT_DIR / "Validation_Sample_Summary.csv",
                                                  index=False, encoding="utf-8-sig")
    overall_metrics.to_csv(OUTPUT_DIR / "Overall_Metrics.csv", index=False,
                           encoding="utf-8-sig")
    zone_metrics.to_csv(OUTPUT_DIR / "Zone_Metrics.csv", index=False,
                        encoding="utf-8-sig")
    auc_confidence.to_csv(OUTPUT_DIR / "AUC_Confidence_Interval.csv", index=False,
                          encoding="utf-8-sig")
    overall_roc.to_csv(OUTPUT_DIR / "ROC_Overall.csv", index=False,
                       encoding="utf-8-sig")
    zone_roc.to_csv(OUTPUT_DIR / "ROC_All_Zones.csv", index=False,
                    encoding="utf-8-sig")

    plot_overall_roc(overall_roc)
    plot_zone_roc(zone_roc)
    plot_confusion_matrices(points)
    plot_metric_comparison(overall_metrics, zone_metrics)

    display = ["Model", "Samples", "ROC_AUC", "PR_AUC_AP", "Accuracy",
               "BalancedAccuracy", "Precision_PPV", "Recall_Sensitivity",
               "Specificity", "F1", "BrierScore"]
    print("验证样本清理汇总:")
    print(pd.DataFrame([sample_summary]).to_string(index=False))
    print("\n总体精度:")
    print(overall_metrics[display].to_string(index=False))
    print("\nAUC Bootstrap 95%置信区间:")
    print(auc_confidence.to_string(index=False))
    print(f"\n输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
