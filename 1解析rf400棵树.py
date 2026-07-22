# -*- coding:utf-8 -*-

"""
解析RandomForest结构
适用于sklearn RandomForestClassifier

作者：STRUT-RF
"""

import os
import sys
import types
import joblib
import numpy as np
import pandas as pd

# Smart App Control may block scikit-learn's unsigned HistGradientBoosting
# extension. This script only needs RandomForestClassifier, but importing
# sklearn.ensemble eagerly imports that unrelated extension. A process-local
# placeholder lets joblib load the random forest without weakening Windows.
_hgb_module_name = "sklearn.ensemble._hist_gradient_boosting.gradient_boosting"
if _hgb_module_name not in sys.modules:
    _hgb_module = types.ModuleType(_hgb_module_name)
    _hgb_module.HistGradientBoostingClassifier = type(
        "HistGradientBoostingClassifier", (), {}
    )
    _hgb_module.HistGradientBoostingRegressor = type(
        "HistGradientBoostingRegressor", (), {}
    )
    sys.modules[_hgb_module_name] = _hgb_module

#===========================
# 路径
#===========================

MODEL_PATH = r"E:\Data\Model\RF_Plateau722_3070\rf_model.pkl"

OUTPUT = r"E:\Data\Model\RF_Plateau722_3070\STRUT_rf\TreeParser"

os.makedirs(OUTPUT,exist_ok=True)

#===========================
# 读取模型
#===========================

print("="*60)
print("读取随机森林模型...")
print("="*60)

rf = joblib.load(MODEL_PATH)

print(rf)

print()

print("树数量：",len(rf.estimators_))

print("类别数：",rf.n_classes_)

print("输入变量数：",rf.n_features_in_)

print()

#===========================
# 因子名称
#===========================

feature_names = [

    "elev",
    "slope",
    "aspect",
    "plan_curv",
    "profile_curv",
    "curvature",
    "T",
    "rain",
    "FVC"

]
from sklearn.tree import export_text

print("=" * 60)
print("第一棵树结构（用于验证）")
print("=" * 60)

tree_text = export_text(
    rf.estimators_[0],
    feature_names=feature_names
)

print(tree_text)
#===========================
# 开始解析
#===========================

all_nodes=[]

print("="*60)
print("开始解析400棵树...")
print("="*60)

for tree_id,estimator in enumerate(rf.estimators_):

    tree=estimator.tree_

    n_nodes=tree.node_count

    print(f"Tree {tree_id+1:03d} : {n_nodes} Nodes")

    for node in range(n_nodes):

        feature=tree.feature[node]

        if feature==-2:

            feature_name="Leaf"

        else:

            feature_name=feature_names[feature]

        all_nodes.append({

            "Tree":tree_id,

            "Node":node,

            "FeatureIndex":feature,

            "FeatureName":feature_name,

            "Threshold":tree.threshold[node],

            "LeftChild":tree.children_left[node],

            "RightChild":tree.children_right[node],

            "Samples":tree.n_node_samples[node],

            "Impurity":tree.impurity[node],

            "Value0":tree.value[node][0][0],

            "Value1":tree.value[node][0][1]

        })

#===========================
# 保存CSV
#===========================

df=pd.DataFrame(all_nodes)

csv_path=os.path.join(

OUTPUT,

"RF_Tree_Structure.csv"

)

df.to_csv(

csv_path,

index=False,

encoding="utf-8-sig"

)

print()

print("="*60)

print("解析完成")

print("CSV：",csv_path)

print("总节点数：",len(df))

print("="*60)
