# -*- coding: utf-8 -*-
"""
STRUT-RF 第5步：使用第4步生成的五个分区模型预测滑坡易发性。

输入：
    UpdateModel_corrected_step4/Zone1_STRUT_RF.pkl ... Zone5_STRUT_RF.pkl
    scaler.pkl
    SOM_5.tif
    9个环境因子栅格

输出：
    Prediction_corrected_step5/Zone1_STRUT_RF.tif ... Zone5_STRUT_RF.tif
    Prediction_corrected_step5/STRUT_RF_Final.tif
    Prediction_corrected_step5/Prediction_Summary.csv
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio


ROOT = Path(r"E:\Data\Model\RF_Plateau")
WORK = ROOT / "STRUT_rf"
MODEL_DIR = WORK / "UpdateModel_corrected_step4"
OUTPUT_DIR = WORK / "Prediction_corrected_step5"
SCALER_PATH = ROOT / "scaler.pkl"
SOM_PATH = Path(r"E:\Data\SOFM\SOM_result6\SOM_5.tif")

# 顺序必须与源RF训练、第2步目标样本提取时完全一致。
FACTOR_PATHS = [
    Path(r"E:\Data\Data\DEM_1km_WGS84.tif"),
    Path(r"E:\Data\Data\slope.tif"),
    Path(r"E:\Data\Data\aspect.tif"),
    Path(r"E:\Data\Data\plan_curv.tif"),
    Path(r"E:\Data\Data\profile_curv.tif"),
    Path(r"E:\Data\Data\Curvatu.tif"),
    Path(r"E:\Data\Data\annual_mean_T.tif"),
    Path(r"E:\Data\Data\RAIN_1km_WGS84.tif"),
    Path(r"E:\Data\Data\FVC_1km_WGS84.tif"),
]


def read_factor_stack():
    """读取同网格的9个因子，并生成所有因子共同有效的像元掩膜。"""
    with rasterio.open(FACTOR_PATHS[0]) as src:
        profile = src.profile.copy()
        height = src.height
        width = src.width
        reference_crs = src.crs
        reference_transform = src.transform

    layers = []
    valid_mask = np.ones((height, width), dtype=bool)
    for path in FACTOR_PATHS:
        with rasterio.open(path) as src:
            if (
                src.shape != (height, width)
                or src.crs != reference_crs
                or src.transform != reference_transform
            ):
                raise ValueError(f"因子栅格与参考网格不一致: {path}")
            data = src.read(1, masked=True)
        valid_mask &= ~np.ma.getmaskarray(data)
        valid_mask &= np.isfinite(data.data)
        layers.append(data.filled(np.nan).astype(np.float32))

    factor_stack = np.stack(layers, axis=-1)
    return factor_stack, valid_mask, profile


def read_som(reference_shape, profile):
    """读取SOM分区并确认其网格与因子一致。"""
    with rasterio.open(SOM_PATH) as src:
        if (
            src.shape != reference_shape
            or src.crs != profile["crs"]
            or src.transform != profile["transform"]
        ):
            raise ValueError("SOM栅格与因子参考网格不一致")
        som = src.read(1)
    return som


def probability_statistics(zone, probability):
    return {
        "Zone": zone,
        "Pixels": int(len(probability)),
        "Minimum": float(np.min(probability)),
        "Median": float(np.median(probability)),
        "P95": float(np.quantile(probability, 0.95)),
        "Maximum": float(np.max(probability)),
    }


def write_tif(output_file, data, profile):
    output_profile = profile.copy()
    output_profile.update(
        dtype=rasterio.float32,
        count=1,
        nodata=np.nan,
        compress="lzw",
    )
    with rasterio.open(output_file, "w", **output_profile) as dst:
        dst.write(data.astype(np.float32), 1)


def predict_zone(
    zone,
    model,
    scaler,
    factor_stack,
    factor_valid_mask,
    som,
    positive_index,
):
    """提取一个分区的像元，标准化后预测滑坡类别概率。"""
    result = np.full(som.shape, np.nan, dtype=np.float32)
    mask = (som == zone) & factor_valid_mask
    pixel_count = int(mask.sum())
    if pixel_count == 0:
        return result, np.empty(0, dtype=np.float32)

    x = factor_stack[mask]
    x = np.nan_to_num(x)

    # 必须使用与源RF训练和第2步目标样本完全相同的变换。
    x = scaler.transform(x)
    probability = model.predict_proba(x)[:, positive_index]

    if not np.all(np.isfinite(probability)):
        raise ValueError(f"Zone {zone}: 预测概率中存在 NaN/Inf")
    if probability.min() < -1e-12 or probability.max() > 1 + 1e-12:
        raise ValueError(
            f"Zone {zone}: 概率超出[0,1]，"
            f"范围={probability.min()}..{probability.max()}"
        )

    result[mask] = probability.astype(np.float32)
    return result, probability


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scaler = joblib.load(SCALER_PATH)
    factor_stack, factor_valid_mask, profile = read_factor_stack()
    som = read_som(factor_stack.shape[:2], profile)

    print(f"因子栅格形状: {factor_stack.shape}")
    print(f"SOM分区: {np.unique(som)}")
    invalid_in_zones = np.isin(som, [1, 2, 3, 4, 5]) & ~factor_valid_mask
    print(f"分区内因子NoData像元: {invalid_in_zones.sum()}")

    final_map = np.full(som.shape, np.nan, dtype=np.float32)
    summaries = []

    for zone in range(1, 6):
        model_file = MODEL_DIR / f"Zone{zone}_STRUT_RF.pkl"
        if not model_file.exists():
            raise FileNotFoundError(f"找不到第四步模型: {model_file}")

        model = joblib.load(model_file)
        positive_positions = np.flatnonzero(model.classes_ == 1)
        if len(positive_positions) != 1:
            raise ValueError(
                f"Zone {zone}: 模型类别 {model.classes_} 中没有唯一的滑坡类别1"
            )
        positive_index = int(positive_positions[0])

        zone_map, probability = predict_zone(
            zone,
            model,
            scaler,
            factor_stack,
            factor_valid_mask,
            som,
            positive_index,
        )
        valid = np.isfinite(zone_map)
        final_map[valid] = zone_map[valid]

        zone_output = OUTPUT_DIR / f"Zone{zone}_STRUT_RF.tif"
        write_tif(zone_output, zone_map, profile)
        summary = probability_statistics(zone, probability)
        summaries.append(summary)
        print(
            f"Zone {zone}: pixels={summary['Pixels']}, "
            f"range={summary['Minimum']:.6f}..{summary['Maximum']:.6f}, "
            f"median={summary['Median']:.6f}"
        )
        print(f"保存: {zone_output}")

    final_output = OUTPUT_DIR / "STRUT_RF_Final.tif"
    write_tif(final_output, final_map, profile)

    valid_final = final_map[np.isfinite(final_map)]
    summaries.append(probability_statistics("Final", valid_final))
    summary_file = OUTPUT_DIR / "Prediction_Summary.csv"
    pd.DataFrame(summaries).to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"最终结果: {final_output}")
    print(
        f"最终范围={valid_final.min():.6f}..{valid_final.max():.6f}, "
        f"中位数={np.median(valid_final):.6f}"
    )
    print(f"统计文件: {summary_file}")


if __name__ == "__main__":
    main()
