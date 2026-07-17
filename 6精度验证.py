# -*- coding: utf-8 -*-
"""
STRUT-RF 精度验证：按 SOM 分区进行分层 K 折交叉验证。

为什么不能直接用最终模型验证 TargetSamples：
最终 Zone1-5 STRUT 模型已经使用了每个分区的全部目标样本。如果再用相同样本
计算精度，会产生数据泄漏和过高的精度。

本程序的验证方式：
1. 每个 Zone 单独进行 StratifiedKFold；
2. 每一折只用训练折目标样本执行完整 STRUT；
3. 测试折完全不参与该折的阈值选择、样本路由、叶分布更新和剪枝；
4. 汇总所有测试折的 OOF（out-of-fold）概率；
5. 在相同测试样本上比较源 RF 与 STRUT-RF。

注意：
该方法能避免“目标域 STRUT 更新”阶段的数据泄漏，但仍需确认目标样本没有参与
最初源 RF 的训练。最严格的最终评价仍应使用两个模型均未见过的独立验证点。

输出：
    AccuracyValidation_CV/OOF_Predictions.csv
    AccuracyValidation_CV/Fold_Metrics.csv
    AccuracyValidation_CV/Overall_Metrics.csv
    AccuracyValidation_CV/Zone_Metrics.csv
    AccuracyValidation_CV/AUC_Comparison.csv
    AccuracyValidation_CV/ROC_Curves.csv
    AccuracyValidation_CV/PR_Curves.csv
    AccuracyValidation_CV/Calibration_Curves.csv
    AccuracyValidation_CV/ROC_PR_Calibration.png
    AccuracyValidation_CV/ROC_All_Zones.png
    AccuracyValidation_CV/Overall_Confusion_Matrices.png
    AccuracyValidation_CV/STRUT_Confusion_Matrices_By_Zone.png
    AccuracyValidation_CV/Overall_Metrics_Comparison.png
    AccuracyValidation_CV/Zone_Metrics_Comparison.png
"""

from pathlib import Path
import argparse
import os

import joblib
import numpy as np
import pandas as pd

# 将 Matplotlib 缓存放在当前可写工作区，避免系统用户目录权限警告。
MPL_CONFIG_DIR = Path(
    r"E:\Data\Model\RF_Plateau\STRUT_rf\.matplotlib"
)
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
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
from sklearn.model_selection import StratifiedKFold

from strut_corrected import adapt_forest


ROOT = Path(r"E:\Data\Model\RF_Plateau")
WORK = ROOT / "STRUT_rf"
SOURCE_MODEL = ROOT / "rf_model.pkl"
TARGET_DIR = WORK / "TargetSamples"
OUTPUT_DIR = WORK / "AccuracyValidation_CV"

DEFAULT_FOLDS = 5
RANDOM_SEED = 20260717
BOOTSTRAP_ITERATIONS = 1000


def positive_class_index(model):
    positions = np.flatnonzero(model.classes_ == 1)
    if len(positions) != 1:
        raise ValueError(f"模型类别 {model.classes_} 中没有唯一的类别1")
    return int(positions[0])


def calculate_metrics(model_name, scope, zone, fold, y_true, probability):
    """固定0.5阈值下的分类指标，以及阈值无关概率指标。"""
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, prediction, labels=[0, 1]
    ).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    fpr, tpr, _ = roc_curve(y_true, probability)

    return {
        "Scope": scope,
        "Zone": zone,
        "Fold": fold,
        "Model": model_name,
        "Threshold": 0.5,
        "Samples": int(len(y_true)),
        "PositiveSamples": int(np.sum(y_true == 1)),
        "NegativeSamples": int(np.sum(y_true == 0)),
        "ROC_AUC": float(roc_auc_score(y_true, probability)),
        "PR_AUC_AP": float(average_precision_score(y_true, probability)),
        "KS": float(np.max(tpr - fpr)),
        "Accuracy": float(accuracy_score(y_true, prediction)),
        "BalancedAccuracy": float(
            balanced_accuracy_score(y_true, prediction)
        ),
        "Precision_PPV": float(
            precision_score(y_true, prediction, zero_division=0)
        ),
        "Recall_Sensitivity": float(
            recall_score(y_true, prediction, zero_division=0)
        ),
        "Specificity": float(specificity),
        "NegativePredictiveValue": float(npv),
        "F1": float(f1_score(y_true, prediction, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, prediction)),
        "CohenKappa": float(cohen_kappa_score(y_true, prediction)),
        "BrierScore": float(brier_score_loss(y_true, probability)),
        "LogLoss": float(log_loss(y_true, probability, labels=[0, 1])),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def bootstrap_auc_comparison(y_true, source_probability, strut_probability):
    """
    分层、配对 bootstrap：
    输出两个模型的 AUC 置信区间及 STRUT - Source 的 AUC 差异区间。
    """
    rng = np.random.default_rng(RANDOM_SEED)
    positive = np.flatnonzero(y_true == 1)
    negative = np.flatnonzero(y_true == 0)
    source_auc = np.empty(BOOTSTRAP_ITERATIONS, dtype=float)
    strut_auc = np.empty(BOOTSTRAP_ITERATIONS, dtype=float)

    for i in range(BOOTSTRAP_ITERATIONS):
        selected_positive = rng.choice(
            positive, size=len(positive), replace=True
        )
        selected_negative = rng.choice(
            negative, size=len(negative), replace=True
        )
        selected = np.r_[selected_positive, selected_negative]
        labels = y_true[selected]
        source_auc[i] = roc_auc_score(
            labels, source_probability[selected]
        )
        strut_auc[i] = roc_auc_score(
            labels, strut_probability[selected]
        )

    difference = strut_auc - source_auc
    return pd.DataFrame(
        [
            {
                "Comparison": "Source_RF",
                "AUC": float(roc_auc_score(y_true, source_probability)),
                "CI95_Lower": float(np.quantile(source_auc, 0.025)),
                "CI95_Upper": float(np.quantile(source_auc, 0.975)),
            },
            {
                "Comparison": "STRUT_RF",
                "AUC": float(roc_auc_score(y_true, strut_probability)),
                "CI95_Lower": float(np.quantile(strut_auc, 0.025)),
                "CI95_Upper": float(np.quantile(strut_auc, 0.975)),
            },
            {
                "Comparison": "STRUT_minus_Source",
                "AUC": float(
                    roc_auc_score(y_true, strut_probability)
                    - roc_auc_score(y_true, source_probability)
                ),
                "CI95_Lower": float(np.quantile(difference, 0.025)),
                "CI95_Upper": float(np.quantile(difference, 0.975)),
            },
        ]
    )


def build_curve_tables(model_name, y_true, probability):
    fpr, tpr, roc_threshold = roc_curve(y_true, probability)
    precision, recall, pr_threshold = precision_recall_curve(
        y_true, probability
    )
    observed, predicted = calibration_curve(
        y_true,
        probability,
        n_bins=10,
        strategy="quantile",
    )

    roc_table = pd.DataFrame(
        {
            "Model": model_name,
            "FPR": fpr,
            "TPR": tpr,
            "Threshold": roc_threshold,
        }
    )
    pr_table = pd.DataFrame(
        {
            "Model": model_name,
            "Recall": recall,
            "Precision": precision,
            "Threshold": np.r_[pr_threshold, np.nan],
        }
    )
    calibration_table = pd.DataFrame(
        {
            "Model": model_name,
            "MeanPredictedProbability": predicted,
            "ObservedPositiveFraction": observed,
        }
    )
    return roc_table, pr_table, calibration_table


def draw_curves(results, roc_tables, pr_tables, calibration_tables):
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.2), dpi=180)

    for model_name in results:
        roc_table = roc_tables[model_name]
        pr_table = pr_tables[model_name]
        calibration_table = calibration_tables[model_name]
        axes[0].plot(
            roc_table["FPR"],
            roc_table["TPR"],
            linewidth=2,
            label=f"{model_name} (AUC={results[model_name]['auc']:.3f})",
        )
        axes[1].plot(
            pr_table["Recall"],
            pr_table["Precision"],
            linewidth=2,
            label=f"{model_name} (AP={results[model_name]['ap']:.3f})",
        )
        axes[2].plot(
            calibration_table["MeanPredictedProbability"],
            calibration_table["ObservedPositiveFraction"],
            marker="o",
            linewidth=2,
            label=model_name,
        )

    axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[2].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[0].set(
        title="Out-of-fold ROC curves",
        xlabel="False positive rate",
        ylabel="True positive rate",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[1].set(
        title="Out-of-fold precision-recall curves",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[2].set(
        title="Out-of-fold calibration curves",
        xlabel="Mean predicted probability",
        ylabel="Observed positive fraction",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "ROC_PR_Calibration.png",
        bbox_inches="tight",
    )
    plt.close(figure)


def build_zone_roc_table(predictions):
    """生成两个模型、每个分区的 ROC 曲线坐标。"""
    rows = []
    probability_columns = {
        "Source_RF": "Source_RF_Probability",
        "STRUT_RF": "STRUT_RF_Probability",
    }
    for zone, group in predictions.groupby("Zone", sort=True):
        y_true = group["TrueLabel"].to_numpy(dtype=int)
        for model_name, column in probability_columns.items():
            probability = group[column].to_numpy(dtype=float)
            fpr, tpr, thresholds = roc_curve(y_true, probability)
            auc_value = roc_auc_score(y_true, probability)
            for fpr_value, tpr_value, threshold in zip(
                fpr, tpr, thresholds
            ):
                rows.append(
                    {
                        "Zone": int(zone),
                        "Model": model_name,
                        "ROC_AUC": float(auc_value),
                        "FPR": float(fpr_value),
                        "TPR": float(tpr_value),
                        "Threshold": float(threshold),
                    }
                )
    return pd.DataFrame(rows)


def draw_zone_roc_curves(zone_roc_table):
    """
    在同一张图中总体展示 Zone 1-5 ROC。
    左图为源 RF，右图为 STRUT-RF；颜色在两个面板中对应同一个 Zone。
    """
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=180)
    colors = plt.cm.tab10(np.linspace(0, 1, 5))

    for axis, model_name, title in [
        (axes[0], "Source_RF", "Source RF: ROC by zone"),
        (axes[1], "STRUT_RF", "STRUT-RF: ROC by zone"),
    ]:
        model_table = zone_roc_table[
            zone_roc_table["Model"] == model_name
        ]
        for color, (zone, group) in zip(
            colors,
            model_table.groupby("Zone", sort=True),
        ):
            auc_value = float(group["ROC_AUC"].iloc[0])
            axis.plot(
                group["FPR"],
                group["TPR"],
                color=color,
                linewidth=2,
                label=f"Zone {int(zone)} (AUC={auc_value:.3f})",
            )
        axis.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        axis.set(
            title=title,
            xlabel="False positive rate",
            ylabel="True positive rate",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right", fontsize=9)

    figure.suptitle(
        "Out-of-fold ROC curves for all SOM zones",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "ROC_All_Zones.png",
        bbox_inches="tight",
    )
    plt.close(figure)


def draw_confusion_matrix(axis, matrix, title):
    """绘制包含计数和按真实类别行归一化百分比的混淆矩阵。"""
    image = axis.imshow(matrix, cmap="Blues", vmin=0)
    row_totals = matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}\n"
                f"({percentages[row, column]:.1%})",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
                fontsize=10,
            )
    axis.set(
        title=title,
        xlabel="Predicted label",
        ylabel="True label",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Non-landslide", "Landslide"],
        yticklabels=["Non-landslide", "Landslide"],
    )
    return image


def draw_overall_confusion_matrices(predictions):
    """绘制源 RF 和 STRUT-RF 的总体 OOF 混淆矩阵。"""
    y_true = predictions["TrueLabel"].to_numpy(dtype=int)
    probability_columns = {
        "Source RF": "Source_RF_Probability",
        "STRUT-RF": "STRUT_RF_Probability",
    }
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=180)
    for axis, (model_name, column) in zip(
        axes, probability_columns.items()
    ):
        prediction = (
            predictions[column].to_numpy(dtype=float) >= 0.5
        ).astype(int)
        matrix = confusion_matrix(
            y_true,
            prediction,
            labels=[0, 1],
        )
        draw_confusion_matrix(
            axis,
            matrix,
            f"{model_name} overall OOF (threshold=0.5)",
        )
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "Overall_Confusion_Matrices.png",
        bbox_inches="tight",
    )
    plt.close(figure)


def draw_strut_zone_confusion_matrices(predictions):
    """在一张图中展示 Zone 1-5 的 STRUT-RF 混淆矩阵。"""
    zones = sorted(predictions["Zone"].unique())
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=180)
    axes = axes.ravel()
    for axis, zone in zip(axes, zones):
        group = predictions[predictions["Zone"] == zone]
        y_true = group["TrueLabel"].to_numpy(dtype=int)
        prediction = (
            group["STRUT_RF_Probability"].to_numpy(dtype=float) >= 0.5
        ).astype(int)
        matrix = confusion_matrix(
            y_true,
            prediction,
            labels=[0, 1],
        )
        draw_confusion_matrix(
            axis,
            matrix,
            f"STRUT-RF Zone {int(zone)}",
        )
    for axis in axes[len(zones):]:
        axis.axis("off")
    figure.suptitle(
        "STRUT-RF out-of-fold confusion matrices by zone "
        "(threshold=0.5)",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "STRUT_Confusion_Matrices_By_Zone.png",
        bbox_inches="tight",
    )
    plt.close(figure)


def draw_overall_metrics_comparison(overall_metrics):
    """绘制总体判别、分类和概率误差指标。"""
    order = ["Source_RF", "STRUT_RF"]
    table = overall_metrics.set_index("Model").loc[order]
    classification_metrics = [
        "ROC_AUC",
        "PR_AUC_AP",
        "Accuracy",
        "BalancedAccuracy",
        "Precision_PPV",
        "Recall_Sensitivity",
        "Specificity",
        "F1",
    ]
    error_metrics = ["BrierScore", "LogLoss"]

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(13, 9),
        dpi=180,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    x = np.arange(len(classification_metrics))
    width = 0.36
    for i, model_name in enumerate(order):
        values = table.loc[model_name, classification_metrics].to_numpy(
            dtype=float
        )
        axes[0].bar(
            x + (i - 0.5) * width,
            values,
            width,
            label=model_name,
        )
    axes[0].set(
        title="Overall OOF discrimination and classification metrics",
        ylabel="Metric value",
        xticks=x,
        xticklabels=classification_metrics,
        ylim=(0, 1),
    )
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    x_error = np.arange(len(error_metrics))
    for i, model_name in enumerate(order):
        values = table.loc[model_name, error_metrics].to_numpy(dtype=float)
        axes[1].bar(
            x_error + (i - 0.5) * width,
            values,
            width,
            label=model_name,
        )
    axes[1].set(
        title="Overall OOF probability error metrics (lower is better)",
        ylabel="Error",
        xticks=x_error,
        xticklabels=error_metrics,
    )
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "Overall_Metrics_Comparison.png",
        bbox_inches="tight",
    )
    plt.close(figure)


def draw_zone_metrics_comparison(zone_metrics):
    """绘制 Zone 1-5 的 AUC、平衡准确率、F1 和 Brier 对比。"""
    metrics = [
        ("ROC_AUC", "ROC-AUC", False),
        ("BalancedAccuracy", "Balanced accuracy", False),
        ("F1", "F1 score", False),
        ("BrierScore", "Brier score (lower is better)", True),
    ]
    zones = sorted(zone_metrics["Zone"].astype(int).unique())
    models = ["Source_RF", "STRUT_RF"]
    width = 0.36
    x = np.arange(len(zones))
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=180)

    for axis, (column, title, is_error) in zip(axes.ravel(), metrics):
        for i, model_name in enumerate(models):
            group = (
                zone_metrics[zone_metrics["Model"] == model_name]
                .set_index("Zone")
                .loc[zones]
            )
            values = group[column].to_numpy(dtype=float)
            axis.bar(
                x + (i - 0.5) * width,
                values,
                width,
                label=model_name,
            )
        axis.set(
            title=title,
            xticks=x,
            xticklabels=[f"Zone {zone}" for zone in zones],
        )
        if not is_error:
            axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()

    figure.suptitle(
        "Out-of-fold metrics by SOM zone",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "Zone_Metrics_Comparison.png",
        bbox_inches="tight",
    )
    plt.close(figure)


def load_zone_data(zone, source_classes):
    target_file = TARGET_DIR / f"Zone_{zone}_NodeSamples.pkl"
    if not target_file.exists():
        raise FileNotFoundError(f"找不到目标样本: {target_file}")
    target = joblib.load(target_file)
    x = np.asarray(target["X"])
    original_y = np.asarray(target["y"])
    y = np.searchsorted(source_classes, original_y).astype(int)
    if x.ndim != 2:
        raise ValueError(f"Zone {zone}: X维度异常 {x.shape}")
    if len(x) != len(y):
        raise ValueError(f"Zone {zone}: X和y样本数量不一致")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"Zone {zone}: 必须同时包含类别0和类别1")
    return x, y


def run_cross_validation(source_rf, zones, n_splits):
    source_index = positive_class_index(source_rf)
    prediction_rows = []
    fold_metric_rows = []

    for zone in zones:
        x, y = load_zone_data(zone, source_rf.classes_)
        class_counts = np.bincount(y, minlength=2)
        if class_counts.min() < n_splits:
            raise ValueError(
                f"Zone {zone}: 最少类别只有 {class_counts.min()} 个样本，"
                f"不能进行 {n_splits} 折分层交叉验证"
            )

        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_SEED + zone,
        )
        print(
            f"\nZone {zone}: samples={len(y)}, class0={class_counts[0]}, "
            f"class1={class_counts[1]}, folds={n_splits}"
        )

        for fold, (train_indices, test_indices) in enumerate(
            splitter.split(x, y), start=1
        ):
            print(
                f"  Fold {fold}/{n_splits}: "
                f"train={len(train_indices)}, test={len(test_indices)}"
            )
            strut_model = adapt_forest(
                source_rf,
                x[train_indices],
                y[train_indices],
            )
            strut_index = positive_class_index(strut_model)
            source_probability = source_rf.predict_proba(
                x[test_indices]
            )[:, source_index]
            strut_probability = strut_model.predict_proba(
                x[test_indices]
            )[:, strut_index]
            test_y = y[test_indices]

            fold_metric_rows.append(
                calculate_metrics(
                    "Source_RF",
                    "Fold",
                    zone,
                    fold,
                    test_y,
                    source_probability,
                )
            )
            fold_metric_rows.append(
                calculate_metrics(
                    "STRUT_RF",
                    "Fold",
                    zone,
                    fold,
                    test_y,
                    strut_probability,
                )
            )

            for local_position, sample_index in enumerate(test_indices):
                prediction_rows.append(
                    {
                        "Zone": zone,
                        "Fold": fold,
                        "ZoneSampleIndex": int(sample_index),
                        "TrueLabel": int(test_y[local_position]),
                        "Source_RF_Probability": float(
                            source_probability[local_position]
                        ),
                        "STRUT_RF_Probability": float(
                            strut_probability[local_position]
                        ),
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    # 每个目标样本在 K 折中必须且只能作为一次测试样本。
    duplicated = predictions.duplicated(
        ["Zone", "ZoneSampleIndex"], keep=False
    )
    if duplicated.any():
        raise ValueError("OOF预测中出现重复测试样本")
    return predictions, pd.DataFrame(fold_metric_rows)


def aggregate_metrics(predictions):
    overall_rows = []
    zone_rows = []
    y_all = predictions["TrueLabel"].to_numpy(dtype=int)

    probability_columns = {
        "Source_RF": "Source_RF_Probability",
        "STRUT_RF": "STRUT_RF_Probability",
    }
    for model_name, column in probability_columns.items():
        probability = predictions[column].to_numpy(dtype=float)
        overall_rows.append(
            calculate_metrics(
                model_name,
                "Overall_OOF",
                "All",
                "All",
                y_all,
                probability,
            )
        )

    for zone, group in predictions.groupby("Zone", sort=True):
        y_zone = group["TrueLabel"].to_numpy(dtype=int)
        for model_name, column in probability_columns.items():
            probability = group[column].to_numpy(dtype=float)
            zone_rows.append(
                calculate_metrics(
                    model_name,
                    "Zone_OOF",
                    int(zone),
                    "All",
                    y_zone,
                    probability,
                )
            )
    return pd.DataFrame(overall_rows), pd.DataFrame(zone_rows)


def main():
    parser = argparse.ArgumentParser(
        description="STRUT-RF zoned stratified cross-validation"
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=DEFAULT_FOLDS,
        help="分层交叉验证折数，默认5",
    )
    parser.add_argument(
        "--zones",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="要验证的分区，默认1 2 3 4 5",
    )
    args = parser.parse_args()
    if args.folds < 2:
        raise ValueError("交叉验证折数必须至少为2")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_rf = joblib.load(SOURCE_MODEL)
    predictions, fold_metrics = run_cross_validation(
        source_rf,
        args.zones,
        args.folds,
    )
    overall_metrics, zone_metrics = aggregate_metrics(predictions)

    predictions.to_csv(
        OUTPUT_DIR / "OOF_Predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fold_metrics.to_csv(
        OUTPUT_DIR / "Fold_Metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    overall_metrics.to_csv(
        OUTPUT_DIR / "Overall_Metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    zone_metrics.to_csv(
        OUTPUT_DIR / "Zone_Metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    y_true = predictions["TrueLabel"].to_numpy(dtype=int)
    source_probability = predictions[
        "Source_RF_Probability"
    ].to_numpy(dtype=float)
    strut_probability = predictions[
        "STRUT_RF_Probability"
    ].to_numpy(dtype=float)
    comparison = bootstrap_auc_comparison(
        y_true,
        source_probability,
        strut_probability,
    )
    comparison.to_csv(
        OUTPUT_DIR / "AUC_Comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    results = {}
    roc_tables = {}
    pr_tables = {}
    calibration_tables = {}
    for model_name, probability in {
        "Source_RF": source_probability,
        "STRUT_RF": strut_probability,
    }.items():
        roc_table, pr_table, calibration_table = build_curve_tables(
            model_name,
            y_true,
            probability,
        )
        roc_tables[model_name] = roc_table
        pr_tables[model_name] = pr_table
        calibration_tables[model_name] = calibration_table
        results[model_name] = {
            "auc": roc_auc_score(y_true, probability),
            "ap": average_precision_score(y_true, probability),
        }

    pd.concat(roc_tables.values(), ignore_index=True).to_csv(
        OUTPUT_DIR / "ROC_Curves.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(pr_tables.values(), ignore_index=True).to_csv(
        OUTPUT_DIR / "PR_Curves.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(calibration_tables.values(), ignore_index=True).to_csv(
        OUTPUT_DIR / "Calibration_Curves.csv",
        index=False,
        encoding="utf-8-sig",
    )
    zone_roc_table = build_zone_roc_table(predictions)
    zone_roc_table.to_csv(
        OUTPUT_DIR / "Zone_ROC_Curves.csv",
        index=False,
        encoding="utf-8-sig",
    )
    draw_curves(
        results,
        roc_tables,
        pr_tables,
        calibration_tables,
    )
    draw_zone_roc_curves(zone_roc_table)
    draw_overall_confusion_matrices(predictions)
    draw_strut_zone_confusion_matrices(predictions)
    draw_overall_metrics_comparison(overall_metrics)
    draw_zone_metrics_comparison(zone_metrics)

    display_columns = [
        "Model",
        "Samples",
        "ROC_AUC",
        "PR_AUC_AP",
        "Accuracy",
        "BalancedAccuracy",
        "Precision_PPV",
        "Recall_Sensitivity",
        "Specificity",
        "F1",
        "BrierScore",
    ]
    print("\nOOF总体精度:")
    print(overall_metrics[display_columns].to_string(index=False))
    print("\nAUC配对Bootstrap比较:")
    print(comparison.to_string(index=False))
    print(f"\n输出目录: {OUTPUT_DIR}")
    print(
        "注意：结果未使用最终全样本模型本身，而是使用每折只在训练折上"
        "重建的STRUT模型，这是避免目标域验证泄漏所必需的。"
    )


if __name__ == "__main__":
    main()
