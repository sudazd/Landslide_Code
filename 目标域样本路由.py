# -*- coding:utf-8 -*-

"""
STRUT-RF
Step 2:
Target Domain Sample Routing

功能:
1.读取源域RF模型
2.读取SOM五分区
3.提取目标域滑坡/非滑坡样本
4.将目标域样本送入400棵源RF树
5.记录每个Node对应目标域样本

输出:
Zone1_NodeSamples.pkl
...
Zone5_NodeSamples.pkl

"""

import os
import joblib
import numpy as np
import rasterio
import geopandas as gpd

from rasterio.enums import Resampling


# ==============================
# 1.路径
# ==============================

RF_MODEL = r"E:\Data\Model\RF_Plateau\rf_model.pkl"

SCALER = r"E:\Data\Model\RF_Plateau\scaler.pkl"


SOM_PATH = r"E:\Data\SOFM\SOM_result6\SOM_5.tif"


landslide_shp = r"E:\Data\Data\landslide_zone_250.shp"

nonlandslide_shp = r"E:\Data\Data\non_landslide_zone250.shp"



factor_paths={


"elev": r"E:\Data\Data\DEM_1km_WGS84.tif",
    "slope": r"E:\Data\Data\slope.tif",
    "aspect": r"E:\Data\Data\aspect.tif",
    "plan_curv": r"E:\Data\Data\plan_curv.tif",
    "profile_curv": r"E:\Data\Data\profile_curv.tif",
    "curvature": r"E:\Data\Data\Curvatu.tif",
    "T": r"E:\Data\Data\annual_mean_T.tif",
    "rain": r"E:\Data\Data\RAIN_1km_WGS84.tif",
    "FVC":r"E:\Data\Data\FVC_1km_WGS84.tif"

}


OUTPUT=r"E:\Data\Model\RF_Plateau\STRUT_rf\TargetSamples"

os.makedirs(
    OUTPUT,
    exist_ok=True
)



# ==============================
# 2.读取RF
# ==============================


print("读取RF模型")

rf=joblib.load(RF_MODEL)


print(
"树数量:",
len(rf.estimators_)
)



scaler=joblib.load(SCALER)



# ==============================
# 3.读取因子
# ==============================


print("读取因子")


ref_path=list(factor_paths.values())[0]


with rasterio.open(ref_path) as src:

    ref_profile=src.profile

    ref_shape=(

        src.height,

        src.width

    )


factor_stack=[]


for name,path in factor_paths.items():

    with rasterio.open(path) as src:

        data=src.read(
            1,
            out_shape=ref_shape,
            resampling=Resampling.bilinear
        )

        factor_stack.append(
            data.astype(np.float32)
        )



factor_stack=np.stack(
    factor_stack,
    axis=-1
)


rows,cols,nf=factor_stack.shape



# ==============================
# 4.读取SOM
# ==============================


print("读取SOM")


with rasterio.open(SOM_PATH) as src:

    som=src.read(1)



print(
"分区:",
np.unique(som)
)



# ==============================
# 5.点提取因子
# ==============================


def extract_points(shp):


    gdf=gpd.read_file(shp)

    gdf=gdf.to_crs(
        ref_profile["crs"]
    )


    samples=[]

    coords=[]


    for geom in gdf.geometry:

        x=geom.x
        y=geom.y


        col,row=~ref_profile["transform"]*(x,y)

        row=int(row)
        col=int(col)


        if (

            row>=0
            and row<rows
            and col>=0
            and col<cols

        ):

            samples.append(
                factor_stack[row,col,:]
            )

            coords.append(
                (row,col)
            )


    return np.array(samples),coords




print("提取滑坡点")


X_ls,coord_ls=extract_points(
    landslide_shp
)


print(
"滑坡:",
X_ls.shape
)



print("提取非滑坡点")


X_nls,coord_nls=extract_points(
    nonlandslide_shp
)


print(
"非滑坡:",
X_nls.shape
)



# 标签

y_ls=np.ones(
    len(X_ls)
)


y_nls=np.zeros(
    len(X_nls)
)



X=np.vstack(
    [
        X_ls,
        X_nls
    ]
)


y=np.hstack(
    [
        y_ls,
        y_nls
    ]
)


coords=coord_ls+coord_nls



# ==============================
# 6.标准化
# ==============================


X=np.nan_to_num(X)


X=scaler.transform(
    X
)



print(
"目标域样本:",
X.shape
)



# ==============================
# 7.按照SOM分区划分
# ==============================


zone_samples={}


for idx,(row,col) in enumerate(coords):


    zone=som[row,col]


    if zone not in zone_samples:

        zone_samples[zone]=[]


    zone_samples[zone].append(idx)



print(
"区域数量:",
len(zone_samples)
)



# ==============================
# 8.Node路由函数
# ==============================


def route_samples(tree,X):


    """

    返回：

    node_id:

        sample index list


    """


    node_map={}


    decision=tree.decision_path(X)


    for sid in range(X.shape[0]):


        start=decision.indptr[sid]

        end=decision.indptr[sid+1]


        nodes=decision.indices[start:end]


        for node in nodes:


            if node not in node_map:

                node_map[node]=[]


            node_map[node].append(
                sid
            )


    return node_map





# ==============================
# 9.五个分区分别路由
# ==============================


print(
"开始STRUT样本路由"
)



for zone,ids in zone_samples.items():


    print(
    "\n处理区域:",
    zone
    )


    X_zone=X[ids]

    y_zone=y[ids]



    forest_map={}



    for tid,tree in enumerate(rf.estimators_):


        print(
        "Tree:",
        tid
        )


        node_map=route_samples(
            tree,
            X_zone
        )


        forest_map[tid]=node_map



    save=os.path.join(

        OUTPUT,

        f"Zone_{zone}_NodeSamples.pkl"

    )



    joblib.dump(

        {

        "X":X_zone,

        "y":y_zone,

        "node_map":forest_map

        },

        save

    )



    print(
    "保存:",
    save
    )



print("===================")

print("全部完成")

print("===================")