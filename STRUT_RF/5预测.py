# -*- coding:utf-8 -*-

"""
STRUT-RF
Step5:
Landslide Susceptibility Mapping

功能:
1.读取Zone1-5 STRUT_RF模型
2.读取SOM_5分区
3.每个分区使用对应模型预测
4.输出Zone susceptibility tif
5.融合最终结果


"""


import os
import joblib
import numpy as np
import rasterio

from rasterio.enums import Resampling



# ==================================
# 1.路径设置
# ==================================

SCALER = r"E:\Data\Model\RF_Plateau722_3070\scaler.pkl"
scaler = joblib.load(SCALER)
MODEL_DIR = (
r"E:\Data\Model\RF_Plateau722_3070\STRUT_rf\UpdateModel_corrected_step4"
)


SOM_PATH = (
r"E:\Data\SOFM\SOM_result6\SOM_5.tif"
)



FACTOR_PATHS = {


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



OUTPUT_DIR = (
r"E:\Data\Model\RF_Plateau722_3070\STRUT_rf\Prediction2"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)



# ==================================
# 2.读取参考栅格
# ==================================


ref_path=list(
    FACTOR_PATHS.values()
)[0]


with rasterio.open(ref_path) as src:

    profile=src.profile

    height=src.height

    width=src.width



# ==================================
# 3.读取所有因子
# ==================================


print("读取因子...")


factor_list=[]


for name,path in FACTOR_PATHS.items():


    with rasterio.open(path) as src:


        data=src.read(
            1,
            out_shape=(
                height,
                width
            ),
            resampling=Resampling.bilinear
        )


        factor_list.append(
            data.astype(np.float32)
        )



factor_stack=np.stack(
    factor_list,
    axis=-1
)


print(
"因子:",
factor_stack.shape
)



# ==================================
# 4.读取SOM
# ==================================


with rasterio.open(SOM_PATH) as src:

    som=src.read(1)



print(
"SOM分区:",
np.unique(som)
)



# ==================================
# 5.预测函数
# ==================================


def predict_zone(
        zone_id,
        model
):


    print(
        "预测Zone:",
        zone_id
    )


    result=np.full(
        (height,width),
        np.nan,
        dtype=np.float32
    )


    mask_zone=(
        som==zone_id
    )


    rows,cols=np.where(
        mask_zone
    )


    print(
        "像元数量:",
        len(rows)
    )


    if len(rows)==0:
        return result



    # 提取区域因子

    X=factor_stack[
        rows,
        cols,
        :
    ]



    X=np.nan_to_num(
        X
    )



    # RF概率预测
    X = np.nan_to_num(X)
    X = scaler.transform(X)

    prob = model.predict_proba(X)[:, 1]
    prob=model.predict_proba(
        X
    )[:,1]



    result[
        rows,
        cols
    ]=prob



    return result





# ==================================
# 6.五个区域预测
# ==================================


zone_results=[]



for zone in range(1,6):


    model_path=os.path.join(

        MODEL_DIR,

        f"Zone{zone}_STRUT_RF.pkl"

    )


    print(
        "\n加载模型:",
        model_path
    )


    model=joblib.load(
        model_path
    )


    zone_map=predict_zone(
        zone,
        model
    )


    zone_results.append(
        zone_map
    )


    out=os.path.join(

        OUTPUT_DIR,

        f"Zone{zone}_STRUT_RF.tif"

    )


    out_profile=profile.copy()


    out_profile.update(

        dtype=rasterio.float32,

        nodata=np.nan,

        compress="lzw"

    )


    with rasterio.open(
        out,
        "w",
        **out_profile
    ) as dst:


        dst.write(
            zone_map,
            1
        )


    print(
        "保存:",
        out
    )



# ==================================
# 7.融合五个分区
# ==================================


print("\n融合最终结果")


final_map=np.full(
    (height,width),
    np.nan,
    dtype=np.float32
)



for zone_map in zone_results:


    mask_valid=(
        ~np.isnan(zone_map)
    )


    final_map[
        mask_valid
    ]=zone_map[
        mask_valid
    ]




final_output=os.path.join(

    OUTPUT_DIR,

    "STRUT_RF_Final.tif"

)



profile.update(

    dtype=rasterio.float32,

    nodata=np.nan,

    compress="lzw"

)



with rasterio.open(

    final_output,

    "w",

    **profile

) as dst:


    dst.write(
        final_map,
        1
    )



print("======================")

print("全部完成")

print(final_output)

print("======================")