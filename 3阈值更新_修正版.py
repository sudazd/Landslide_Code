# -*- coding: utf-8 -*-
"""
STRUT-RF 第3步：按照 Segev et al. (2015) 更新阈值。

与旧程序的主要区别：
1. 候选阈值使用相邻不同特征值的中点；
2. 只在 Information Gain (IG) 的局部极大值中选择阈值；
3. 使用 Jensen-Shannon Divergence Gain (DG) 保持源树左右分布；
4. 同时计算左右分布反转，反转更优时记录 Swap=True；
5. 采用自顶向下递归，祖先阈值更新后立即重新路由后代样本；
6. 同时输出不可达节点和叶节点的目标域类别分布，供第4步使用。

输入：
    rf_model.pkl
    TargetSamples/Zone_1_NodeSamples.pkl ... Zone_5_NodeSamples.pkl

输出：
    Threshold_corrected/Zone1_STRUT_Operations.csv ... Zone5_STRUT_Operations.csv
    Threshold_corrected/STRUT_Update_Summary.csv
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(r"E:\Data\Model\RF_Plateau722_3070")
WORK = ROOT / "STRUT_rf"
RF_MODEL = ROOT / "rf_model.pkl"
TARGET_DIR = WORK / "TargetSamples"
OUTPUT_DIR = WORK / "Threshold_corrected"


def normalize(values):
    """将类别计数或权重转换为概率分布。"""
    values = np.asarray(values, dtype=float).ravel()
    total = values.sum()
    if total <= 0:
        return np.full(len(values), 1.0 / len(values))
    return values / total


def entropy(counts):
    """以2为底的 Shannon entropy。"""
    p = normalize(counts)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def js_divergence(p, q):
    """论文公式(2)中的 Jensen-Shannon divergence。"""
    p = normalize(p)
    q = normalize(q)
    m = 0.5 * (p + q)

    def kl(a, b):
        valid = a > 0
        return float(np.sum(a[valid] * np.log2(a[valid] / b[valid])))

    return 0.5 * (kl(p, m) + kl(q, m))


def calculate_candidates(values, labels, n_classes):
    """计算中点候选阈值、IG、左右类别计数。"""
    order = np.argsort(values, kind="mergesort")
    sorted_values = np.asarray(values)[order]
    sorted_labels = np.asarray(labels, dtype=int)[order]

    # 只有相邻特征值不同时，二者中点才是有效分割阈值。
    split_positions = np.flatnonzero(sorted_values[:-1] < sorted_values[1:])
    if len(split_positions) == 0:
        return None

    thresholds = (
        sorted_values[split_positions]
        + (sorted_values[split_positions + 1] - sorted_values[split_positions]) / 2.0
    )

    one_hot = np.eye(n_classes, dtype=float)[sorted_labels]
    cumulative = np.cumsum(one_hot, axis=0)
    left_counts = cumulative[split_positions]
    total_counts = cumulative[-1]
    right_counts = total_counts - left_counts

    parent_entropy = entropy(total_counts)
    n_samples = len(sorted_labels)
    information_gain = np.empty(len(split_positions), dtype=float)

    for i, position in enumerate(split_positions):
        n_left = position + 1
        n_right = n_samples - n_left
        information_gain[i] = (
            parent_entropy
            - (n_left / n_samples) * entropy(left_counts[i])
            - (n_right / n_samples) * entropy(right_counts[i])
        )

    return thresholds, information_gain, left_counts, right_counts


def local_ig_maxima(information_gain):
    """论文公式(3)中的 IG 局部极大值约束。"""
    left_neighbor = np.r_[-np.inf, information_gain[:-1]]
    right_neighbor = np.r_[information_gain[1:], -np.inf]
    return (information_gain >= left_neighbor) & (information_gain >= right_neighbor)


def divergence_gain(left_counts, right_counts, source_left, source_right):
    """论文公式(1)的 DG。"""
    n_left = left_counts.sum()
    n_right = right_counts.sum()
    n_total = n_left + n_right
    return float(
        1.0
        - (n_left / n_total) * js_divergence(left_counts, source_left)
        - (n_right / n_total) * js_divergence(right_counts, source_right)
    )


def best_orientation(candidates, source_left, source_right):
    """分别计算正常方向和左右反转方向的最优阈值。"""
    thresholds, information_gain, left_counts, right_counts = candidates
    eligible = np.flatnonzero(local_ig_maxima(information_gain))

    def choose(q_left, q_right):
        scores = np.array(
            [
                divergence_gain(left_counts[i], right_counts[i], q_left, q_right)
                for i in eligible
            ],
            dtype=float,
        )
        best_position = int(np.argmax(scores))
        candidate_index = int(eligible[best_position])
        return {
            "threshold": float(thresholds[candidate_index]),
            "dg": float(scores[best_position]),
            "ig": float(information_gain[candidate_index]),
            "candidate_count": int(len(thresholds)),
            "local_max_count": int(len(eligible)),
        }

    normal = choose(source_left, source_right)
    swapped = choose(source_right, source_left)
    return normal, swapped


def update_one_tree(tree_id, estimator, x, y, n_classes):
    """自顶向下计算一棵树的所有 STRUT 操作。"""
    tree = estimator.tree_
    working_left = tree.children_left.copy()
    working_right = tree.children_right.copy()
    operations = []

    def visit(node_id, sample_ids, depth):
        sample_ids = np.asarray(sample_ids, dtype=int)

        # Algorithm 2: 没有目标样本到达时，记录剪枝操作。
        if len(sample_ids) == 0:
            operations.append(
                {
                    "Tree": tree_id,
                    "Node": node_id,
                    "Depth": depth,
                    "Action": "Prune",
                    "FeatureIndex": int(tree.feature[node_id]),
                    "OldThreshold": float(tree.threshold[node_id]),
                    "NewThreshold": np.nan,
                    "Swap": False,
                    "DGNormal": np.nan,
                    "DGSwapped": np.nan,
                    "SelectedDG": np.nan,
                    "SelectedIG": np.nan,
                    "CandidateNumber": 0,
                    "LocalMaximumNumber": 0,
                    "SampleNumber": 0,
                    "NewLeftChild": -1,
                    "NewRightChild": -1,
                    "TargetValue0": np.nan,
                    "TargetValue1": np.nan,
                }
            )
            return

        # Algorithm 2: 到达源树叶节点时，记录目标域叶分布。
        if tree.children_left[node_id] == -1:
            counts = np.bincount(y[sample_ids], minlength=n_classes).astype(float)
            probs = counts / counts.sum()
            operations.append(
                {
                    "Tree": tree_id,
                    "Node": node_id,
                    "Depth": depth,
                    "Action": "LeafUpdate",
                    "FeatureIndex": -2,
                    "OldThreshold": float(tree.threshold[node_id]),
                    "NewThreshold": np.nan,
                    "Swap": False,
                    "DGNormal": np.nan,
                    "DGSwapped": np.nan,
                    "SelectedDG": np.nan,
                    "SelectedIG": np.nan,
                    "CandidateNumber": 0,
                    "LocalMaximumNumber": 0,
                    "SampleNumber": int(len(sample_ids)),
                    "NewLeftChild": -1,
                    "NewRightChild": -1,
                    "TargetValue0": float(probs[0]),
                    "TargetValue1": float(probs[1]),
                }
            )
            return

        feature_index = int(tree.feature[node_id])
        old_threshold = float(tree.threshold[node_id])
        original_left = int(tree.children_left[node_id])
        original_right = int(tree.children_right[node_id])
        source_left = normalize(tree.value[original_left, 0, :])
        source_right = normalize(tree.value[original_right, 0, :])
        values = x[sample_ids, feature_index]
        candidates = calculate_candidates(values, y[sample_ids], n_classes)

        if candidates is None:
            # 当前节点的目标样本特征值完全相同，保留源阈值。
            new_threshold = old_threshold
            swap = False
            dg_normal = np.nan
            dg_swapped = np.nan
            selected_dg = np.nan
            selected_ig = np.nan
            candidate_number = 0
            local_maximum_number = 0
        else:
            normal, swapped = best_orientation(candidates, source_left, source_right)
            swap = swapped["dg"] > normal["dg"]
            selected = swapped if swap else normal
            new_threshold = selected["threshold"]
            dg_normal = normal["dg"]
            dg_swapped = swapped["dg"]
            selected_dg = selected["dg"]
            selected_ig = selected["ig"]
            candidate_number = selected["candidate_count"]
            local_maximum_number = selected["local_max_count"]

        if swap:
            working_left[node_id], working_right[node_id] = (
                working_right[node_id],
                working_left[node_id],
            )

        new_left = int(working_left[node_id])
        new_right = int(working_right[node_id])
        operations.append(
            {
                "Tree": tree_id,
                "Node": node_id,
                "Depth": depth,
                "Action": "ThresholdUpdate",
                "FeatureIndex": feature_index,
                "OldThreshold": old_threshold,
                "NewThreshold": new_threshold,
                "Swap": swap,
                "DGNormal": dg_normal,
                "DGSwapped": dg_swapped,
                "SelectedDG": selected_dg,
                "SelectedIG": selected_ig,
                "CandidateNumber": candidate_number,
                "LocalMaximumNumber": local_maximum_number,
                "SampleNumber": int(len(sample_ids)),
                "NewLeftChild": new_left,
                "NewRightChild": new_right,
                "TargetValue0": np.nan,
                "TargetValue1": np.nan,
            }
        )

        goes_left = values <= new_threshold
        visit(new_left, sample_ids[goes_left], depth + 1)
        visit(new_right, sample_ids[~goes_left], depth + 1)

    visit(0, np.arange(len(x), dtype=int), 0)
    return operations


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rf = joblib.load(RF_MODEL)
    n_classes = len(rf.classes_)
    summaries = []

    for zone in range(1, 6):
        target_file = TARGET_DIR / f"Zone_{zone}_NodeSamples.pkl"
        target = joblib.load(target_file)
        x = np.asarray(target["X"])
        original_y = np.asarray(target["y"])
        # sklearn 树内部类别位置为 0..K-1。
        y = np.searchsorted(rf.classes_, original_y).astype(int)

        all_operations = []
        for tree_id, estimator in enumerate(rf.estimators_):
            all_operations.extend(
                update_one_tree(tree_id, estimator, x, y, n_classes)
            )

        result = pd.DataFrame(all_operations)
        output_file = OUTPUT_DIR / f"Zone{zone}_STRUT_Operations.csv"
        result.to_csv(output_file, index=False, encoding="utf-8-sig")

        action_counts = result["Action"].value_counts()
        threshold_rows = result[result["Action"] == "ThresholdUpdate"]
        summary = {
            "Zone": zone,
            "TargetSamples": len(x),
            "ThresholdUpdates": int(action_counts.get("ThresholdUpdate", 0)),
            "Swaps": int(threshold_rows["Swap"].sum()),
            "LeafUpdates": int(action_counts.get("LeafUpdate", 0)),
            "Prunes": int(action_counts.get("Prune", 0)),
            "MeanAbsThresholdChange": float(
                np.mean(np.abs(threshold_rows["NewThreshold"] - threshold_rows["OldThreshold"]))
            ),
            "Output": str(output_file),
        }
        summaries.append(summary)
        print(
            f"Zone {zone}: samples={summary['TargetSamples']}, "
            f"thresholds={summary['ThresholdUpdates']}, swaps={summary['Swaps']}, "
            f"leaves={summary['LeafUpdates']}, prunes={summary['Prunes']}"
        )
        print(f"saved: {output_file}")

    summary_df = pd.DataFrame(summaries)
    summary_file = OUTPUT_DIR / "STRUT_Update_Summary.csv"
    summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")
    print(f"summary: {summary_file}")


if __name__ == "__main__":
    main()
