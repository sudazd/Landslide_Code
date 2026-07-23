"""黄土高原五分区 PU-Bagging 非滑坡点筛选。

处理流程：
1. 读取整个黄土高原的全部因子栅格；
2. 读取 zone1.shp ... zone5.shp 五个分区面；
3. 对每个分区读取对应滑坡点，生成未标记样本并独立运行 PU-Bagging；
4. 在滑坡点缓冲区外筛选低正类概率点，数量与该区滑坡点完全相同；
5. 输出 nolandslide_zone1.shp ... nolandslide_zone5.shp；
6. 合并五个分区，输出 nolandslide_loess_plateau.shp。

使用前优先修改下面“用户配置区”的路径、分区值和文件命名规则。
也可以使用命令行参数覆盖。
因子文件夹中应只放参与模型训练的 .tif/.tiff 因子栅格。
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import bounds as window_bounds
from shapely.geometry import mapping
from shapely.geometry import Point
from sklearn.tree import DecisionTreeClassifier


# =============================================================================
# 用户配置区：请根据实际数据修改
# =============================================================================
# 整个黄土高原因子栅格所在文件夹；程序自动读取其中全部 tif/tiff。
FACTOR_DIR = Path(r"E:\Data\Data")

# 五个分区面文件所在文件夹和命名规则。
ZONE_POLYGON_DIR = Path(r"E:\Data\zone_landslide\zone_polygon")
ZONE_POLYGON_TEMPLATE = "zone{zone}.shp"

# 依次处理的分区栅格值。
ZONE_VALUES = ["1", "2", "3", "4", "5"]

# 为抵消因子NoData，先从每个分区面内随机保留未标记样本数的若干倍候选像元。
ZONE_CANDIDATE_MULTIPLIER = 2.0

# 五个分区滑坡点文件所在文件夹及命名模板。
# 例如：landslide_zone1.shp、landslide_zone2.shp ...
LANDSLIDE_DIR = Path(r"E:\Data\zone_landslide\landslide_zone")
LANDSLIDE_TEMPLATE = "landslide_zone{zone}.shp"

# 输出文件夹。
OUTPUT_DIR = Path(r"E:\Data\nonlandslide\nolandslide_zone")

# 每个分区生成的未标记样本数量。
UNLABELED_COUNT = 80_000

# PU-Bagging 参数。
N_ESTIMATORS = 200
TREE_MAX_DEPTH = 6
PU_THRESHOLD = 0.5

# 非滑坡点必须位于所有滑坡点该距离以外；投影坐标系单位应为米。
BUFFER_DISTANCE = 500.0

# 仅用于按“米”计算缓冲距离；因子和分区面可以继续使用经纬度坐标系。
# 这里沿用原始参考程序的 UTM 49N。输出点仍保持因子栅格的原坐标系。
BUFFER_CRS = "EPSG:32649"

# 固定随机种子，使重复运行结果一致。
RANDOM_SEED = 20260723


def union_all(geometries: gpd.GeoSeries):
    """兼容不同版本 GeoPandas 的几何合并方法。"""
    if hasattr(geometries, "union_all"):
        return geometries.union_all()
    return geometries.unary_union


def discover_factor_paths(factor_dir: Path) -> list[Path]:
    """发现因子文件夹中的全部 GeoTIFF。"""
    if not factor_dir.is_dir():
        raise FileNotFoundError(f"找不到因子文件夹：{factor_dir}")

    factor_paths = sorted(
        list(factor_dir.glob("*.tif")) + list(factor_dir.glob("*.tiff")),
        key=lambda path: path.name.lower(),
    )
    if not factor_paths:
        raise FileNotFoundError(f"因子文件夹内没有 tif/tiff：{factor_dir}")
    return factor_paths


def open_and_validate_factors(
    stack: ExitStack, factor_paths: list[Path]
) -> tuple[list[rasterio.io.DatasetReader], tuple[float, float, float, float]]:
    """打开因子栅格，验证坐标系并计算共同覆盖范围。"""
    datasets = [stack.enter_context(rasterio.open(path)) for path in factor_paths]
    reference = datasets[0]
    if reference.crs is None:
        raise ValueError(f"因子没有坐标系：{factor_paths[0]}")
    if not np.isclose(reference.transform.b, 0.0) or not np.isclose(
        reference.transform.d, 0.0
    ):
        raise ValueError("参考因子栅格存在旋转，当前像元抽样方法不支持旋转栅格。")

    for path, dataset in zip(factor_paths, datasets):
        if dataset.count < 1:
            raise ValueError(f"因子没有有效波段：{path}")
        if dataset.crs is None:
            raise ValueError(f"因子没有坐标系：{path}")
        if dataset.crs != reference.crs:
            raise ValueError(
                f"因子坐标系不一致：{path.name}。请先将全部因子统一投影。"
            )

    left = max(dataset.bounds.left for dataset in datasets)
    bottom = max(dataset.bounds.bottom for dataset in datasets)
    right = min(dataset.bounds.right for dataset in datasets)
    top = min(dataset.bounds.top for dataset in datasets)
    if left >= right or bottom >= top:
        raise ValueError("所有因子栅格之间不存在共同覆盖范围。")

    print(f"发现{len(datasets)}个因子栅格：")
    for path in factor_paths:
        print(f"  - {path.name}")
    print(f"因子坐标系：{reference.crs}")
    if not reference.crs.is_projected:
        print("注意：因子为地理坐标系；程序将在缓冲筛选时临时转换为米制投影。")
    return datasets, (left, bottom, right, top)


def sample_factor_features(
    datasets: list[rasterio.io.DatasetReader], points: list[Point]
) -> tuple[np.ndarray, np.ndarray]:
    """在点位置提取全部因子值，并返回特征和有效行掩膜。"""
    if not points:
        return np.empty((0, len(datasets)), dtype=np.float32), np.zeros(0, bool)

    coordinates = [(point.x, point.y) for point in points]
    features = np.full((len(points), len(datasets)), np.nan, dtype=np.float32)

    for factor_index, dataset in enumerate(datasets):
        sampled = list(dataset.sample(coordinates, indexes=1, masked=True))
        column = np.ma.vstack(sampled).reshape(len(points), -1)[:, 0]
        if np.ma.isMaskedArray(column):
            values = column.astype(np.float64).filled(np.nan)
        else:
            values = np.asarray(column)
        features[:, factor_index] = np.asarray(values, dtype=np.float32)

    valid = np.isfinite(features).all(axis=1)
    return features, valid


def load_zone_polygon(path: Path, target_crs, zone_value: str):
    """读取一个分区面、投影到因子坐标系并合并为单一几何。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到分区{zone_value}面文件：{path}")

    zone_gdf = gpd.read_file(path)
    if zone_gdf.empty:
        raise ValueError(f"分区{zone_value}面文件为空：{path}")
    if zone_gdf.crs is None:
        raise ValueError(f"分区{zone_value}面文件没有坐标系：{path}")
    if not zone_gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise TypeError(f"分区{zone_value}必须全部为面或多面几何。")

    zone_gdf = zone_gdf.to_crs(target_crs)
    geometry = union_all(zone_gdf.geometry)
    if geometry is None or geometry.is_empty:
        raise ValueError(f"分区{zone_value}几何为空。")
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    if geometry.is_empty:
        raise ValueError(f"分区{zone_value}几何修复后仍为空。")
    return geometry


def load_positive_points(
    path: Path,
    target_crs,
    zone_geometry,
    datasets: list[rasterio.io.DatasetReader],
    zone_value: str,
) -> tuple[gpd.GeoDataFrame, np.ndarray]:
    """读取并严格验证一个分区的滑坡点及其因子值。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到分区{zone_value}的滑坡点：{path}")

    positives = gpd.read_file(path)
    if positives.empty:
        raise ValueError(f"分区{zone_value}的滑坡点文件为空：{path}")
    if positives.crs is None:
        raise ValueError(f"分区{zone_value}的滑坡点没有坐标系：{path}")
    if not positives.geometry.geom_type.eq("Point").all():
        raise TypeError(f"分区{zone_value}的滑坡数据必须全部为点几何。")

    positives = positives.to_crs(target_crs).copy().reset_index(drop=True)
    inside_zone = positives.geometry.apply(zone_geometry.covers).to_numpy()
    if not inside_zone.all():
        invalid_count = int((~inside_zone).sum())
        raise ValueError(
            f"分区{zone_value}有{invalid_count}个滑坡点不在对应分区面内。"
            "请检查分区面、坐标系或滑坡点文件。"
        )

    positive_features, valid = sample_factor_features(
        datasets, positives.geometry.tolist()
    )
    if not valid.all():
        invalid_count = int((~valid).sum())
        raise ValueError(
            f"分区{zone_value}有{invalid_count}个滑坡点位于因子范围外、NoData区，"
            "或包含非有限因子值。为保证正负样本数量严格一致，请先修正这些点。"
        )

    return positives, positive_features


def generate_unlabeled_samples(
    reference: rasterio.io.DatasetReader,
    datasets: list[rasterio.io.DatasetReader],
    zone_geometry,
    common_bounds: tuple[float, float, float, float],
    sample_count: int,
    candidate_multiplier: float,
    rng: np.random.Generator,
    zone_value: str,
) -> tuple[gpd.GeoDataFrame, np.ndarray]:
    """从分区面内的唯一参考栅格像元中心生成有效未标记样本。"""
    if sample_count <= 0:
        raise ValueError("未标记样本数量必须大于0。")
    if candidate_multiplier < 1.0:
        raise ValueError("分区候选倍数必须大于或等于1。")

    left = max(zone_geometry.bounds[0], common_bounds[0])
    bottom = max(zone_geometry.bounds[1], common_bounds[1])
    right = min(zone_geometry.bounds[2], common_bounds[2])
    top = min(zone_geometry.bounds[3], common_bounds[3])
    if left >= right or bottom >= top:
        raise ValueError(f"分区{zone_value}面与全部因子的共同范围没有重叠。")

    reservoir_size = int(np.ceil(sample_count * candidate_multiplier))
    reservoir_keys = np.empty(0, dtype=np.float64)
    reservoir_rows = np.empty(0, dtype=np.int64)
    reservoir_cols = np.empty(0, dtype=np.int64)
    available_zone_cells = 0

    print(
        f"分区{zone_value}：扫描分区面覆盖的参考因子像元，"
        f"随机保留最多{reservoir_size}个候选像元……"
    )
    transform = reference.transform
    geometry_mapping = [mapping(zone_geometry)]
    for _, window in reference.block_windows(1):
        block_left, block_bottom, block_right, block_top = window_bounds(
            window, transform
        )
        if (
            block_right < left
            or block_left > right
            or block_top < bottom
            or block_bottom > top
        ):
            continue

        inside_polygon = geometry_mask(
            geometry_mapping,
            out_shape=(int(window.height), int(window.width)),
            transform=reference.window_transform(window),
            all_touched=False,
            invert=True,
        )
        local_rows, local_cols = np.nonzero(inside_polygon)
        if len(local_rows) == 0:
            continue

        rows = local_rows.astype(np.int64) + int(window.row_off)
        cols = local_cols.astype(np.int64) + int(window.col_off)
        centre_cols = cols.astype(np.float64) + 0.5
        centre_rows = rows.astype(np.float64) + 0.5
        x_coords = (
            transform.a * centre_cols
            + transform.b * centre_rows
            + transform.c
        )
        y_coords = (
            transform.d * centre_cols
            + transform.e * centre_rows
            + transform.f
        )
        inside_factor_bounds = (
            (x_coords >= left)
            & (x_coords <= right)
            & (y_coords >= bottom)
            & (y_coords <= top)
        )
        rows = rows[inside_factor_bounds]
        cols = cols[inside_factor_bounds]
        if len(rows) == 0:
            continue

        available_zone_cells += len(rows)
        block_keys = rng.random(len(rows))
        combined_keys = np.concatenate([reservoir_keys, block_keys])
        combined_rows = np.concatenate([reservoir_rows, rows])
        combined_cols = np.concatenate([reservoir_cols, cols])

        if len(combined_keys) > reservoir_size:
            keep = np.argpartition(combined_keys, reservoir_size - 1)[:reservoir_size]
            reservoir_keys = combined_keys[keep]
            reservoir_rows = combined_rows[keep]
            reservoir_cols = combined_cols[keep]
        else:
            reservoir_keys = combined_keys
            reservoir_rows = combined_rows
            reservoir_cols = combined_cols

    if available_zone_cells < sample_count:
        raise ValueError(
            f"分区{zone_value}面内且位于因子共同范围的参考像元只有"
            f"{available_zone_cells}个，少于所需{sample_count}个未标记样本。"
        )

    centre_cols = reservoir_cols.astype(np.float64) + 0.5
    centre_rows = reservoir_rows.astype(np.float64) + 0.5
    x_coords = (
        transform.a * centre_cols + transform.b * centre_rows + transform.c
    )
    y_coords = (
        transform.d * centre_cols + transform.e * centre_rows + transform.f
    )
    candidate_points = [
        Point(x_coord, y_coord)
        for x_coord, y_coord in zip(x_coords.tolist(), y_coords.tolist())
    ]
    candidate_features, valid = sample_factor_features(datasets, candidate_points)
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) < sample_count:
        raise ValueError(
            f"分区{zone_value}从{len(candidate_points)}个分区像元候选中只得到"
            f"{len(valid_indices)}个全部因子有效的点，少于{sample_count}个。"
            "请增大 ZONE_CANDIDATE_MULTIPLIER，或检查因子的NoData范围。"
        )

    chosen = rng.choice(valid_indices, sample_count, replace=False)
    accepted_points = [candidate_points[index] for index in chosen]
    features = candidate_features[chosen].astype(np.float32, copy=False)
    unlabeled = gpd.GeoDataFrame(geometry=accepted_points, crs=reference.crs)
    print(f"分区{zone_value}：未标记样本生成完成。")
    return unlabeled, features


def pu_bagging(
    positive_features: np.ndarray,
    unlabeled_features: np.ndarray,
    n_estimators: int,
    max_depth: int,
    rng: np.random.Generator,
    zone_value: str,
) -> np.ndarray:
    """对一个分区执行基于袋外预测的 PU-Bagging。"""
    positive_count = len(positive_features)
    unlabeled_count = len(unlabeled_features)
    if positive_count == 0 or unlabeled_count == 0:
        raise ValueError(f"分区{zone_value}的正样本或未标记样本为空。")
    if n_estimators <= 0 or max_depth <= 0:
        raise ValueError("模型数量和决策树深度必须大于0。")
    if unlabeled_count <= positive_count:
        raise ValueError(
            f"分区{zone_value}的未标记样本数必须大于滑坡点数。"
            "请增大 UNLABELED_COUNT。"
        )

    scores = np.zeros(unlabeled_count, dtype=np.float64)
    counts = np.zeros(unlabeled_count, dtype=np.int32)
    print(f"分区{zone_value}：开始PU-Bagging……")

    for estimator_index in range(n_estimators):
        bootstrap_indices = rng.choice(
            unlabeled_count, positive_count, replace=True
        )
        training_features = np.vstack(
            [positive_features, unlabeled_features[bootstrap_indices]]
        )
        training_labels = np.concatenate(
            [
                np.ones(positive_count, dtype=np.uint8),
                np.zeros(positive_count, dtype=np.uint8),
            ]
        )

        classifier = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
        )
        classifier.fit(training_features, training_labels)

        out_of_bag = np.ones(unlabeled_count, dtype=bool)
        out_of_bag[np.unique(bootstrap_indices)] = False
        if out_of_bag.any():
            probabilities = classifier.predict_proba(
                unlabeled_features[out_of_bag]
            )[:, 1]
            scores[out_of_bag] += probabilities
            counts[out_of_bag] += 1

        completed = estimator_index + 1
        if completed % 50 == 0 or completed == n_estimators:
            print(f"  已完成：{completed}/{n_estimators}")

    never_evaluated = counts == 0
    if never_evaluated.any():
        fallback_indices = rng.choice(
            unlabeled_count, positive_count, replace=True
        )
        fallback_features = np.vstack(
            [positive_features, unlabeled_features[fallback_indices]]
        )
        fallback_labels = np.concatenate(
            [
                np.ones(positive_count, dtype=np.uint8),
                np.zeros(positive_count, dtype=np.uint8),
            ]
        )
        fallback = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
        fallback.fit(fallback_features, fallback_labels)
        scores[never_evaluated] = fallback.predict_proba(
            unlabeled_features[never_evaluated]
        )[:, 1]
        counts[never_evaluated] = 1

    print(f"分区{zone_value}：PU-Bagging完成。")
    return scores / counts


def select_negative_points(
    unlabeled: gpd.GeoDataFrame,
    scores: np.ndarray,
    positives: gpd.GeoDataFrame,
    required_count: int,
    threshold: float,
    buffer_distance: float,
    buffer_crs,
    zone_value: str,
) -> gpd.GeoDataFrame:
    """筛选低概率、缓冲区外的唯一非滑坡点。"""
    # 距离运算必须在米制投影中进行，但输出仍保留因子原坐标系。
    positives_metric = positives.to_crs(buffer_crs)
    unlabeled_metric = unlabeled.to_crs(buffer_crs)
    buffer_geometry = union_all(
        positives_metric.geometry.buffer(buffer_distance)
    )
    outside_buffer = ~unlabeled_metric.geometry.intersects(
        buffer_geometry
    ).to_numpy()
    candidate_indices = np.flatnonzero((scores < threshold) & outside_buffer)

    if len(candidate_indices) < required_count:
        outside_count = int(outside_buffer.sum())
        raise ValueError(
            f"分区{zone_value}需要{required_count}个非滑坡点，但概率<{threshold}且"
            f"位于{buffer_distance}米缓冲区外的候选点只有{len(candidate_indices)}个"
            f"（缓冲区外总数：{outside_count}）。请增大 UNLABELED_COUNT、提高"
            "PU_THRESHOLD，或适当减小 BUFFER_DISTANCE。"
        )

    # 选择PU正类概率最低的点，确保是候选池中最可靠的非滑坡样本。
    ranked_indices = candidate_indices[
        np.argsort(scores[candidate_indices], kind="stable")
    ]
    selected_indices = ranked_indices[:required_count]
    selected = unlabeled.iloc[selected_indices].copy().reset_index(drop=True)
    selected["label"] = np.int16(0)
    selected["zone"] = str(zone_value)
    selected["pu_score"] = scores[selected_indices].astype(np.float32)
    return selected[["geometry", "label", "zone", "pu_score"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按五个分区运行PU-Bagging并输出等量非滑坡点。"
    )
    parser.add_argument("--factor-dir", type=Path, default=FACTOR_DIR)
    parser.add_argument("--zone-polygon-dir", type=Path, default=ZONE_POLYGON_DIR)
    parser.add_argument(
        "--zone-polygon-template", default=ZONE_POLYGON_TEMPLATE
    )
    parser.add_argument("--zone-values", nargs=5, default=ZONE_VALUES)
    parser.add_argument(
        "--zone-candidate-multiplier",
        type=float,
        default=ZONE_CANDIDATE_MULTIPLIER,
    )
    parser.add_argument("--landslide-dir", type=Path, default=LANDSLIDE_DIR)
    parser.add_argument("--landslide-template", default=LANDSLIDE_TEMPLATE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--unlabeled-count", type=int, default=UNLABELED_COUNT)
    parser.add_argument("--estimators", type=int, default=N_ESTIMATORS)
    parser.add_argument("--max-depth", type=int, default=TREE_MAX_DEPTH)
    parser.add_argument("--threshold", type=float, default=PU_THRESHOLD)
    parser.add_argument("--buffer-distance", type=float, default=BUFFER_DISTANCE)
    parser.add_argument("--buffer-crs", default=BUFFER_CRS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.zone_polygon_dir.is_dir():
        raise FileNotFoundError(f"找不到分区面文件夹：{args.zone_polygon_dir}")
    if not args.landslide_dir.is_dir():
        raise FileNotFoundError(f"找不到滑坡点文件夹：{args.landslide_dir}")
    if args.unlabeled_count <= 0:
        raise ValueError("未标记样本数量必须大于0。")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("PU概率阈值必须位于0到1之间。")
    if args.buffer_distance < 0:
        raise ValueError("缓冲距离不能小于0。")
    if args.zone_candidate_multiplier < 1.0:
        raise ValueError("分区候选倍数必须大于或等于1。")
    try:
        buffer_crs = rasterio.crs.CRS.from_user_input(args.buffer_crs)
    except Exception as exc:
        raise ValueError(f"无法识别缓冲投影坐标系：{args.buffer_crs}") from exc
    if not buffer_crs.is_projected:
        raise ValueError("BUFFER_CRS/--buffer-crs 必须是以米为单位的投影坐标系。")

    factor_paths = discover_factor_paths(args.factor_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        datasets, common_bounds = open_and_validate_factors(stack, factor_paths)
        reference = datasets[0]

        zone_results: list[gpd.GeoDataFrame] = []
        summary_rows: list[dict[str, object]] = []

        for zone_index, zone_value in enumerate(args.zone_values, start=1):
            print("=" * 70)
            print(f"开始处理分区{zone_value}（{zone_index}/5）")
            zone_polygon_path = (
                args.zone_polygon_dir
                / args.zone_polygon_template.format(zone=zone_value)
            )
            zone_geometry = load_zone_polygon(
                zone_polygon_path, reference.crs, str(zone_value)
            )
            print(f"分区面：{zone_polygon_path}")
            positive_path = args.landslide_dir / args.landslide_template.format(
                zone=zone_value
            )
            positives, positive_features = load_positive_points(
                positive_path,
                reference.crs,
                zone_geometry,
                datasets,
                str(zone_value),
            )
            positive_count = len(positives)
            print(f"分区{zone_value}：有效滑坡点数量={positive_count}")

            zone_rng = np.random.default_rng(
                np.random.SeedSequence([args.seed, zone_index])
            )
            unlabeled, unlabeled_features = generate_unlabeled_samples(
                reference,
                datasets,
                zone_geometry,
                common_bounds,
                args.unlabeled_count,
                args.zone_candidate_multiplier,
                zone_rng,
                str(zone_value),
            )
            scores = pu_bagging(
                positive_features,
                unlabeled_features,
                args.estimators,
                args.max_depth,
                zone_rng,
                str(zone_value),
            )
            negatives = select_negative_points(
                unlabeled,
                scores,
                positives,
                positive_count,
                args.threshold,
                args.buffer_distance,
                buffer_crs,
                str(zone_value),
            )

            zone_output = args.output_dir / f"nolandslide_zone{zone_value}.shp"
            negatives.to_file(zone_output, index=False, encoding="UTF-8")
            zone_results.append(negatives)
            summary_rows.append(
                {
                    "zone": str(zone_value),
                    "landslide_count": positive_count,
                    "nolandslide_count": len(negatives),
                    "minimum_pu_score": float(negatives["pu_score"].min()),
                    "maximum_pu_score": float(negatives["pu_score"].max()),
                    "output": str(zone_output),
                }
            )
            print(
                f"分区{zone_value}完成：滑坡点={positive_count}，"
                f"非滑坡点={len(negatives)}"
            )
            print(f"输出：{zone_output}")

    merged = gpd.GeoDataFrame(
        pd.concat(zone_results, ignore_index=True),
        geometry="geometry",
        crs=zone_results[0].crs,
    )
    merged_output = args.output_dir / "nolandslide_loess_plateau.shp"
    merged.to_file(merged_output, index=False, encoding="UTF-8")

    summary = pd.DataFrame(summary_rows)
    summary_output = args.output_dir / "pu_bagging_summary.csv"
    summary.to_csv(summary_output, index=False, encoding="utf-8-sig")

    print("=" * 70)
    print("五个分区全部处理完成！")
    print(f"合并非滑坡点总数：{len(merged)}")
    print(f"黄土高原整体输出：{merged_output}")
    print(f"数量与概率汇总：{summary_output}")


if __name__ == "__main__":
    main()
