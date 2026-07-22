# -*- coding: utf-8 -*-
"""
STRUT-RF 第4步：将第3步生成的完整操作写回随机森林。

输入：
    E:/Data/Model/RF_Plateau/rf_model.pkl
    Threshold_corrected/Zone1_STRUT_Operations.csv ... Zone5_STRUT_Operations.csv

处理的操作：
    ThresholdUpdate  更新阈值，并按 NewLeftChild/NewRightChild 设置子树方向
    LeafUpdate       使用目标域类别分布更新叶节点 tree.value
    Prune            将目标域不可达节点变成叶节点

输出：
    UpdateModel_corrected_step4/Zone1_STRUT_RF.pkl ... Zone5_STRUT_RF.pkl
"""

from pathlib import Path
import copy

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import _tree


ROOT = Path(r"E:\Data\Model\RF_Plateau722_3070")
WORK = ROOT / "STRUT_rf"
RF_MODEL = ROOT / "rf_model.pkl"
OPERATIONS_DIR = WORK / "Threshold_corrected"
OUTPUT_DIR = WORK / "UpdateModel_corrected_step4"


def require_columns(dataframe, required, csv_file):
    missing = sorted(set(required) - set(dataframe.columns))
    if missing:
        raise ValueError(f"{csv_file} 缺少字段: {missing}")


def update_threshold(tree, row):
    """应用内部节点阈值和第3步确定的左右子树方向。"""
    node = int(row.Node)
    new_threshold = float(row.NewThreshold)
    new_left = int(row.NewLeftChild)
    new_right = int(row.NewRightChild)

    if not np.isfinite(new_threshold):
        raise ValueError(f"Tree={int(row.Tree)}, Node={node}: NewThreshold 不是有效数值")
    if new_left < 0 or new_right < 0:
        raise ValueError(f"Tree={int(row.Tree)}, Node={node}: 内部节点子节点编号无效")

    tree.threshold[node] = new_threshold
    # 直接使用第3步输出的最终子节点编号，比再次根据 Swap 翻转更稳妥。
    tree.children_left[node] = new_left
    tree.children_right[node] = new_right


def update_leaf(tree, row, n_classes):
    """用目标域样本的经验类别分布更新叶节点预测概率。"""
    node = int(row.Node)
    probabilities = np.array(
        [float(row.TargetValue0), float(row.TargetValue1)],
        dtype=float,
    )
    if len(probabilities) != n_classes:
        raise ValueError("当前脚本用于二分类；模型类别数量不是2")
    if not np.all(np.isfinite(probabilities)) or probabilities.sum() <= 0:
        raise ValueError(f"Tree={int(row.Tree)}, Node={node}: 叶节点目标分布无效")

    probabilities /= probabilities.sum()
    tree.children_left[node] = _tree.TREE_LEAF
    tree.children_right[node] = _tree.TREE_LEAF
    tree.feature[node] = _tree.TREE_UNDEFINED
    tree.threshold[node] = _tree.TREE_UNDEFINED
    tree.value[node, 0, :] = probabilities


def prune_unreachable(tree, row):
    """将没有目标域样本到达的子树根节点变为叶节点。"""
    node = int(row.Node)
    tree.children_left[node] = _tree.TREE_LEAF
    tree.children_right[node] = _tree.TREE_LEAF
    tree.feature[node] = _tree.TREE_UNDEFINED
    tree.threshold[node] = _tree.TREE_UNDEFINED
    # 该节点没有目标样本，无法估计新的类别分布；保留源模型 tree.value。


def validate_tree(tree, tree_id):
    """检查从根节点可达的树结构，防止非法子节点或循环。"""
    visited = set()
    stack = [0]
    while stack:
        node = int(stack.pop())
        if node in visited:
            raise ValueError(f"Tree={tree_id}: 节点 {node} 重复到达，树结构可能存在循环")
        if node < 0 or node >= tree.node_count:
            raise ValueError(f"Tree={tree_id}: 非法节点编号 {node}")
        visited.add(node)

        left = int(tree.children_left[node])
        right = int(tree.children_right[node])
        if left == _tree.TREE_LEAF:
            if right != _tree.TREE_LEAF:
                raise ValueError(f"Tree={tree_id}, Node={node}: 左右叶标志不一致")
            continue
        if right == _tree.TREE_LEAF:
            raise ValueError(f"Tree={tree_id}, Node={node}: 左右叶标志不一致")
        stack.extend([left, right])
    return len(visited)


def update_forest(source_rf, operations, zone):
    """为一个分区生成独立的 STRUT 随机森林。"""
    new_rf = copy.deepcopy(source_rf)
    n_classes = len(new_rf.classes_)
    if n_classes != 2:
        raise ValueError(f"Zone {zone}: 当前代码只支持二分类模型，实际类别数={n_classes}")

    stats = {
        "ThresholdUpdate": 0,
        "LeafUpdate": 0,
        "Prune": 0,
        "ReachableNodes": 0,
    }

    for tree_id, tree_rows in operations.groupby("Tree", sort=True):
        tree_id = int(tree_id)
        if tree_id < 0 or tree_id >= len(new_rf.estimators_):
            raise ValueError(f"Zone {zone}: CSV 中存在非法 Tree={tree_id}")
        tree = new_rf.estimators_[tree_id].tree_

        # CSV 是深度优先顺序。按 Depth 再稳定排序，确保父节点先于子节点应用。
        tree_rows = tree_rows.sort_values("Depth", kind="stable")
        for row in tree_rows.itertuples(index=False):
            action = str(row.Action)
            if action == "ThresholdUpdate":
                update_threshold(tree, row)
            elif action == "LeafUpdate":
                update_leaf(tree, row, n_classes)
            elif action == "Prune":
                prune_unreachable(tree, row)
            else:
                raise ValueError(
                    f"Zone {zone}, Tree={tree_id}, Node={int(row.Node)}: 未知 Action={action}"
                )
            stats[action] += 1

        stats["ReachableNodes"] += validate_tree(tree, tree_id)

    expected_trees = set(range(len(new_rf.estimators_)))
    actual_trees = set(operations["Tree"].astype(int).unique())
    if actual_trees != expected_trees:
        missing = sorted(expected_trees - actual_trees)
        raise ValueError(f"Zone {zone}: 操作文件缺少树: {missing[:10]}")

    return new_rf, stats


def validate_probabilities(model, x, zone):
    """检查模型输出是否为有效二分类概率。"""
    probability = model.predict_proba(x)
    if probability.shape != (len(x), 2):
        raise ValueError(f"Zone {zone}: predict_proba 输出形状异常 {probability.shape}")
    if not np.all(np.isfinite(probability)):
        raise ValueError(f"Zone {zone}: 概率中存在 NaN/Inf")
    if probability.min() < -1e-12 or probability.max() > 1 + 1e-12:
        raise ValueError(
            f"Zone {zone}: 概率超出[0,1]，范围={probability.min()}..{probability.max()}"
        )
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError(f"Zone {zone}: 两个类别概率之和不等于1")
    return probability


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_rf = joblib.load(RF_MODEL)
    summaries = []

    print(f"源随机森林树数量: {len(source_rf.estimators_)}")

    required_columns = {
        "Tree",
        "Node",
        "Depth",
        "Action",
        "NewThreshold",
        "NewLeftChild",
        "NewRightChild",
        "TargetValue0",
        "TargetValue1",
    }

    for zone in range(1, 6):
        csv_file = OPERATIONS_DIR / f"Zone{zone}_STRUT_Operations.csv"
        target_file = WORK / "TargetSamples" / f"Zone_{zone}_NodeSamples.pkl"
        if not csv_file.exists():
            raise FileNotFoundError(f"找不到第3步输出: {csv_file}")
        if not target_file.exists():
            raise FileNotFoundError(f"找不到目标域样本: {target_file}")

        print(f"\n处理 Zone {zone}: {csv_file}")
        # 保留 CSV 中浮点数的往返精度；阈值处的 1e-16 舍入也可能改变等值样本路径。
        operations = pd.read_csv(csv_file, float_precision="round_trip")
        require_columns(operations, required_columns, csv_file)
        new_rf, stats = update_forest(source_rf, operations, zone)

        target = joblib.load(target_file)
        probability = validate_probabilities(new_rf, np.asarray(target["X"]), zone)
        positive_index = int(np.flatnonzero(new_rf.classes_ == 1)[0])
        positive_probability = probability[:, positive_index]

        output_file = OUTPUT_DIR / f"Zone{zone}_STRUT_RF.pkl"
        joblib.dump(new_rf, output_file)
        summaries.append(
            {
                "Zone": zone,
                "Trees": len(new_rf.estimators_),
                **stats,
                "TargetProbabilityMin": float(positive_probability.min()),
                "TargetProbabilityMax": float(positive_probability.max()),
                "Output": str(output_file),
            }
        )
        print(
            f"完成: thresholds={stats['ThresholdUpdate']}, "
            f"leaves={stats['LeafUpdate']}, prunes={stats['Prune']}, "
            f"target_probability={positive_probability.min():.6f}..{positive_probability.max():.6f}"
        )
        print(f"保存: {output_file}")

    summary_file = OUTPUT_DIR / "STRUT_Model_Update_Summary.csv"
    pd.DataFrame(summaries).to_csv(summary_file, index=False, encoding="utf-8-sig")
    print(f"\n汇总: {summary_file}")


if __name__ == "__main__":
    main()
