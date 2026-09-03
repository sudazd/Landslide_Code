# -*- coding: utf-8 -*-
"""计算原始全区随机森林在SOM Zone 1—5中的分区AUC。

说明：
1. 原始RF仍然是一个全区统一模型，没有进行分区训练或迁移；
2. 本程序只把原始RF在30%测试点上的预测按SOM分区分组评价；
3. 为了与STRUT验证口径一致，统一删除重复点、与70%训练集重合点，
   并只保留RF结果和SOM栅格均有效的测试点；
4. 输出总体及各Zone的ROC-AUC、PR-AUC和bootstrap 95%置信区间。
"""

from pathlib import Path
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio


ROOT = Path(r"E:\Data\Model\RF_Plateau827_3070")
WORK = Path(r"E:\Data\Model\RF_Plateau\STRUT_rf")
MPL_DIR = WORK / ".matplotlib"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


RF_RASTER = ROOT / "RF_result.tif"
SOM_RASTER = Path(r"E:\Data\SOFM\SOM_result6\SOM_5.tif")

LANDSLIDE_TEST = Path(
    r"E:\Data\zone_landslide\landslide_7030_Plateau\landslide_Plateau_30.shp"
)
NONLANDSLIDE_TEST = Path(
    r"E:\No_landslide_random\Plateau_nolandslide30.shp"
)
LANDSLIDE_TRAIN = Path(
    r"E:\Data\zone_landslide\landslide_7030_Plateau\landslide_Plateau_70.shp"
)
NONLANDSLIDE_TRAIN = Path(
    r"E:\No_landslide_random\Plateau_nolandslide70.shp"
)

OUTPUT_DIR = WORK / "RF_Zone_AUC"
COORDINATE_DECIMALS = 8
BOOTSTRAP_ITERATIONS = 3000
RANDOM_SEED = 20260902


def check_files():
    paths = [
        RF_RASTER,
        SOM_RASTER,
        LANDSLIDE_TEST,
        NONLANDSLIDE_TEST,
        LANDSLIDE_TRAIN,
        NONLANDSLIDE_TRAIN,
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("以下输入文件不存在:\n" + "\n".join(missing))


def raster_signature(path):
    with rasterio.open(path) as src:
        return src.crs, src.shape, src.transform


def verify_alignment():
    rf_signature = raster_signature(RF_RASTER)
    som_signature = raster_signature(SOM_RASTER)
    if rf_signature != som_signature:
        raise ValueError("原始RF结果与SOM栅格的CRS、尺寸或网格不一致")
    if rf_signature[0] is None:
        raise ValueError("原始RF结果没有坐标系")
    return rf_signature[0]


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
        raise ValueError(f"点数据没有坐标系: {path}")
    gdf = gdf.to_crs(target_crs)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if not np.all(gdf.geometry.geom_type == "Point"):
        raise ValueError(f"样本必须全部为点: {path}")
    gdf["label"] = int(label)
    gdf["sample_source"] = source
    gdf["coord_key"] = coordinate_key(gdf)
    return gdf[["label", "sample_source", "coord_key", "geometry"]]


def build_clean_test_points(target_crs):
    positive = read_points(
        LANDSLIDE_TEST, target_crs, 1, "landslide_test_30"
    )
    negative = read_points(
        NONLANDSLIDE_TEST, target_crs, 0, "nonlandslide_test_30"
    )
    positive_train = read_points(
        LANDSLIDE_TRAIN, target_crs, 1, "landslide_train_70"
    )
    negative_train = read_points(
        NONLANDSLIDE_TRAIN, target_crs, 0, "nonlandslide_train_70"
    )

    summary = {
        "PositiveOriginal": len(positive),
        "NegativeOriginal": len(negative),
    }
    positive_original = len(positive)
    negative_original = len(negative)
    positive = positive.drop_duplicates("coord_key").copy()
    negative = negative.drop_duplicates("coord_key").copy()
    summary["PositiveDuplicatesRemoved"] = positive_original - len(positive)
    summary["NegativeDuplicatesRemoved"] = negative_original - len(negative)

    all_train_keys = set(positive_train["coord_key"]) | set(
        negative_train["coord_key"]
    )
    positive_overlap = positive["coord_key"].isin(all_train_keys)
    negative_overlap = negative["coord_key"].isin(all_train_keys)
    summary["PositiveTrainOverlapRemoved"] = int(positive_overlap.sum())
    summary["NegativeTrainOverlapRemoved"] = int(negative_overlap.sum())
    positive = positive.loc[~positive_overlap].copy()
    negative = negative.loc[~negative_overlap].copy()

    conflicts = set(positive["coord_key"]) & set(negative["coord_key"])
    if conflicts:
        raise ValueError(
            f"有{len(conflicts)}个坐标同时标记为滑坡和非滑坡，请检查数据"
        )

    points = pd.concat([positive, negative], ignore_index=True)
    points.insert(0, "sample_id", np.arange(len(points), dtype=int))
    points["x"] = points.geometry.x
    points["y"] = points.geometry.y
    summary["SamplesAfterCleaning"] = len(points)
    return points, summary


def sample_raster(path, points):
    coordinates = list(zip(points["x"], points["y"]))
    values = np.full(len(points), np.nan, dtype=float)
    with rasterio.open(path) as src:
        bounds = src.bounds
        for index, sample in enumerate(src.sample(coordinates, masked=True)):
            value = sample[0]
            if not np.ma.is_masked(value):
                values[index] = float(value)
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


def attach_rf_probability_and_zone(points, summary):
    rf_probability, rf_valid = sample_raster(RF_RASTER, points)
    som_value, som_valid = sample_raster(SOM_RASTER, points)
    zones = np.rint(som_value)
    som_valid &= np.isin(zones, [1, 2, 3, 4, 5])
    common_valid = rf_valid & som_valid

    summary["RasterInvalidOrOutsideRemoved"] = int((~common_valid).sum())
    points = points.loc[common_valid].copy().reset_index(drop=True)
    probability = rf_probability[common_valid]
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError(
            f"原始RF概率超出[0,1]: {probability.min()}..{probability.max()}"
        )
    points["zone"] = zones[common_valid].astype(int)
    points["RF_probability"] = probability
    summary["FinalSamples"] = len(points)
    summary["FinalPositive"] = int((points["label"] == 1).sum())
    summary["FinalNegative"] = int((points["label"] == 0).sum())
    return points, summary


def stratified_bootstrap_auc(labels, probability, seed):
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(seed)
    bootstrap_values = np.empty(BOOTSTRAP_ITERATIONS, dtype=float)
    for iteration in range(BOOTSTRAP_ITERATIONS):
        selected = np.r_[
            rng.choice(positive, len(positive), replace=True),
            rng.choice(negative, len(negative), replace=True),
        ]
        bootstrap_values[iteration] = roc_auc_score(
            labels[selected], probability[selected]
        )
    return np.quantile(bootstrap_values, [0.025, 0.975])


def calculate_auc_tables(points):
    metric_rows = []
    roc_rows = []
    groups = [("Overall", "All", points)]
    groups.extend(
        ("Zone", int(zone), group)
        for zone, group in points.groupby("zone", sort=True)
    )

    for group_number, (scope, zone, group) in enumerate(groups):
        labels = group["label"].to_numpy(dtype=int)
        probability = group["RF_probability"].to_numpy(dtype=float)
        if len(np.unique(labels)) < 2:
            print(f"警告: {scope} {zone}只有一个类别，无法计算AUC")
            continue
        auc_value = roc_auc_score(labels, probability)
        ci_lower, ci_upper = stratified_bootstrap_auc(
            labels, probability, RANDOM_SEED + group_number
        )
        metric_rows.append(
            {
                "Scope": scope,
                "Zone": zone,
                "Samples": len(group),
                "PositiveSamples": int((labels == 1).sum()),
                "NegativeSamples": int((labels == 0).sum()),
                "ROC_AUC": auc_value,
                "AUC_CI95_Lower": ci_lower,
                "AUC_CI95_Upper": ci_upper,
                "PR_AUC_AP": average_precision_score(labels, probability),
            }
        )
        fpr, tpr, thresholds = roc_curve(labels, probability)
        for false_positive_rate, true_positive_rate, threshold in zip(
            fpr, tpr, thresholds
        ):
            roc_rows.append(
                {
                    "Scope": scope,
                    "Zone": zone,
                    "ROC_AUC": auc_value,
                    "FPR": false_positive_rate,
                    "TPR": true_positive_rate,
                    "Threshold": threshold,
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(roc_rows)


def plot_zone_roc(roc_table):
    zone_table = roc_table[roc_table["Scope"] == "Zone"]
    figure, axis = plt.subplots(figsize=(8, 6), dpi=180)
    colors = plt.cm.tab10(np.linspace(0, 1, 5))
    for color, (zone, group) in zip(
        colors, zone_table.groupby("Zone", sort=True)
    ):
        auc_value = group["ROC_AUC"].iloc[0]
        axis.plot(
            group["FPR"],
            group["TPR"],
            color=color,
            linewidth=2,
            label=f"Zone {zone} (AUC={auc_value:.3f})",
        )
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="Ordinary RF ROC curves for Zone 1-5",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "RF_ROC_All_Zones.png", bbox_inches="tight")
    plt.close(figure)


def main():
    check_files()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_crs = verify_alignment()
    points, sample_summary = build_clean_test_points(target_crs)
    points, sample_summary = attach_rf_probability_and_zone(
        points, sample_summary
    )
    metrics, roc_table = calculate_auc_tables(points)

    points[
        [
            "sample_id",
            "label",
            "sample_source",
            "x",
            "y",
            "zone",
            "RF_probability",
        ]
    ].to_csv(
        OUTPUT_DIR / "RF_Validation_Point_Predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([sample_summary]).to_csv(
        OUTPUT_DIR / "RF_Validation_Sample_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics.to_csv(
        OUTPUT_DIR / "RF_Overall_and_Zone_AUC.csv",
        index=False,
        encoding="utf-8-sig",
    )
    roc_table.to_csv(
        OUTPUT_DIR / "RF_ROC_Overall_and_Zones.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_zone_roc(roc_table)

    print("\n原始全区RF的总体及分区AUC:")
    print(metrics.to_string(index=False))
    print(f"\n输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
