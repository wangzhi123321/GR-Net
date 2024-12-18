import math
import numpy as np
import torch.nn.functional as F
import torch
from osgeo import gdal
from code.hrnet import HRnet
# import torchvision
# from utils.data_loading import BasicDataset
from torchvision import transforms as T

# 读取tif数据集


def readTif(fileName, xoff=0, yoff=0, data_width=0, data_height=0):
    gdal.SetConfigOption('GTIFF_SRS_SOURCE', 'EPSG')
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    #  栅格矩阵的列数
    width = dataset.RasterXSize
    #  栅格矩阵的行数
    height = dataset.RasterYSize
    #  获取数据
    if (data_width == 0 and data_height == 0):
        data_width = width
        data_height = height
    data = dataset.ReadAsArray(xoff, yoff, data_width, data_height)
    geotransform = dataset.GetGeoTransform()  # 获取仿射变换参数
    projection = dataset.GetProjection()  # 获取投影信息
    return data, geotransform, projection


# 保存tif文件函数
def writeTiff(fileName, data, geotransform, projection):
    if 'int8' in data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(data.shape) == 3:
        im_bands, im_height, im_width = data.shape
    elif len(data.shape) == 2:
        data = np.array([data])
        im_bands, im_height, im_width = data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(fileName, int(im_width), int(
        im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(geotransform)  # 写入仿射变换参数
        dataset.SetProjection(projection)  # 写入投影信息
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(data[i])
    del dataset


#  tif裁剪（tif像素数据，裁剪边长）
def TifCroppingArray(img, SideLength):
    #  裁剪链表
    TifArrayReturn = []
    #  列上图像块数目
    ColumnNum = int((img.shape[0] - SideLength * 2) / (300 - SideLength * 2))
    #  行上图像块数目
    RowNum = int((img.shape[1] - SideLength * 2) / (300 - SideLength * 2))
    for i in range(ColumnNum):
        TifArray = []
        for j in range(RowNum):
            cropped = img[i * (300 - SideLength * 2): i * (300 - SideLength * 2) + 300,
                          j * (300 - SideLength * 2): j * (300 - SideLength * 2) + 300]
            TifArray.append(cropped)
        TifArrayReturn.append(TifArray)
    #  考虑到行列会有剩余的情况，向前裁剪一行和一列
    #  向前裁剪最后一列
    for i in range(ColumnNum):
        cropped = img[i * (300 - SideLength * 2): i * (300 - SideLength * 2) + 300,
                      (img.shape[1] - 300): img.shape[1]]
        TifArrayReturn[i].append(cropped)
    #  向前裁剪最后一行
    TifArray = []
    for j in range(RowNum):
        cropped = img[(img.shape[0] - 300): img.shape[0],
                      j * (300 - SideLength * 2): j * (300 - SideLength * 2) + 300]
        TifArray.append(cropped)
    #  向前裁剪右下角
    cropped = img[(img.shape[0] - 300): img.shape[0],
                  (img.shape[1] - 300): img.shape[1]]
    TifArray.append(cropped)
    TifArrayReturn.append(TifArray)
    #  列上的剩余数
    ColumnOver = (img.shape[0] - SideLength *
                  2) % (300 - SideLength * 2) + SideLength
    #  行上的剩余数
    RowOver = (img.shape[1] - SideLength *
               2) % (300 - SideLength * 2) + SideLength
    return TifArrayReturn, RowOver, ColumnOver


#  获得结果矩阵
def Result(shape, TifArray, npyfile, RepetitiveLength, RowOver, ColumnOver):
    result = np.zeros(shape, np.uint8)
    #  j来标记行数
    j = 0
    for i, img in enumerate(npyfile):
        #  最左侧一列特殊考虑，左边的边缘要拼接进去
        if (i % len(TifArray[0]) == 0):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if (j == 0):
                result[0: 300 - RepetitiveLength, 0: 300 - RepetitiveLength] = img[0: 300 -
                                                                                   RepetitiveLength, 0: 300 - RepetitiveLength]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif (j == len(TifArray) - 1):
                #  原来错误的
                # result[shape[0] - ColumnOver : shape[0], 0 : 512 - RepetitiveLength] = img[0 : ColumnOver, 0 : 512 - RepetitiveLength]
                #  后来修改的
                result[shape[0] - ColumnOver - RepetitiveLength: shape[0], 0: 300 - RepetitiveLength] = img[
                    300 - ColumnOver - RepetitiveLength: 300,
                    0: 300 - RepetitiveLength]
            else:
                result[j * (300 - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                    300 - 2 * RepetitiveLength) + RepetitiveLength,
                    0:300 - RepetitiveLength] = img[RepetitiveLength: 300 - RepetitiveLength, 0: 300 - RepetitiveLength]
                #  最右侧一列特殊考虑，右边的边缘要拼接进去
        elif (i % len(TifArray[0]) == len(TifArray[0]) - 1):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if (j == 0):
                result[0: 300 - RepetitiveLength, shape[1] - RowOver: shape[1]] = img[0: 300 - RepetitiveLength,
                                                                                      300 - RowOver: 300]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif (j == len(TifArray) - 1):
                result[shape[0] - ColumnOver: shape[0], shape[1] - RowOver: shape[1]] = img[300 - ColumnOver: 300,
                                                                                            300 - RowOver: 300]
            else:
                result[j * (300 - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                    300 - 2 * RepetitiveLength) + RepetitiveLength,
                    shape[1] - RowOver: shape[1]] = img[RepetitiveLength: 300 - RepetitiveLength, 300 - RowOver: 300]
                #  走完每一行的最右侧，行数+1
            j = j + 1
        #  不是最左侧也不是最右侧的情况
        else:
            #  第一行的要特殊考虑，上边的边缘要考虑进去
            if (j == 0):
                result[0: 300 - RepetitiveLength,
                       (i - j * len(TifArray[0])) * (300 - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(
                           TifArray[0]) + 1) * (300 - 2 * RepetitiveLength) + RepetitiveLength
                       ] = img[0: 300 - RepetitiveLength, RepetitiveLength: 300 - RepetitiveLength]
                #  最后一行的要特殊考虑，下边的边缘要考虑进去
            if (j == len(TifArray) - 1):
                result[shape[0] - ColumnOver: shape[0],
                       (i - j * len(TifArray[0])) * (300 - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(
                           TifArray[0]) + 1) * (300 - 2 * RepetitiveLength) + RepetitiveLength
                       ] = img[300 - ColumnOver: 300, RepetitiveLength: 300 - RepetitiveLength]
            else:
                result[j * (300 - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                    300 - 2 * RepetitiveLength) + RepetitiveLength,
                    (i - j * len(TifArray[0])) * (300 - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(
                        TifArray[0]) + 1) * (300 - 2 * RepetitiveLength) + RepetitiveLength,
                ] = img[RepetitiveLength: 300 - RepetitiveLength, RepetitiveLength: 300 - RepetitiveLength]
    return result


area_perc = 0.5

# file_path = r"pre_data/img_futian/FuTian.TIF"
# ndvi_file_path = r"pre_data/img_futian_ndvi/FuTian_ndvi.TIF"
# model_paths = [
#     r"runs/bubbliiiing/final_epoch@50/checkpoint_epoch50.pth"
# ]
# Result_GR_Path = r"results/FuTian_pred_hrnet/final_epoch@50/FuTian_pred_hrnet_area_perc@0.5_coordinate.TIF"
# Result_Building_Path = r"results/FuTian_pred_hrnet/final_epoch@50/FuTian_pred_hrnet_area_perc@0.5_coordinate_building.TIF"
# RepetitiveLength = int((1 - math.sqrt(area_perc)) * 300 / 2)

# file_path = r"pre_data/img_luohu/luohu_clip.tif"
# ndvi_file_path = r"pre_data/img_luohu_ndvi/luohu_clip_ndvi.TIF"
# model_paths = [
#     r"runs/bubbliiiing/final_epoch@50/checkpoint_epoch50.pth"
# ]
# Result_GR_Path = r"results/LuoHu_pred_hrnet/final_epoch@50_mask_thres@0.9/LuoHu_pred_hrnet_area_perc@0.5_coordinate_clip.TIF"
# Result_Building_Path = r"results/LuoHu_pred_hrnet/final_epoch@50_mask_thres@0.9/LuoHu_pred_hrnet_area_perc@0.5_coordinate_building_clip.TIF"
# # RepetitiveLength = int((1 - math.sqrt(area_perc)) * 300 / 2)

# file_path = r"pre_data/img_nanshan/tile_0.TIF"
# ndvi_file_path = r"pre_data/img_nanshan_ndvi/tile_0_ndvi.TIF"
# model_paths = [
#     r"runs/bubbliiiing/final_epoch@50/checkpoint_epoch50.pth"
# ]
# Result_GR_Path = r"results/NanShan_pred_hrnet/final_epoch@50_mask_thres@0.9/shenzhen_pred_hrnet.TIF"
# Result_Building_Path = r"results/NanShan_pred_hrnet/final_epoch@50_mask_thres@0.9/shenzhen_pred_hrnet_building.TIF"

file_path = r"pre_data/img_shenzhen/shenzhen_clip.tif"
ndvi_file_path = r"pre_data/img_shenzhen_ndvi/shenzhen_clip_ndvi.TIF"
model_paths = [
    r"runs/bubbliiiing/final_epoch@50/checkpoint_epoch50.pth"
]
Result_GR_Path = r"results/shenzhen_pred/final_epoch@50_mask_thres@0.9/shenzhen_pred_hrnet_clip.TIF"
Result_Building_Path = r"results/shenzhen_pred/final_epoch@50_mask_thres@0.9/shenzhen_pred_hrnet_building_clip.TIF"

RepetitiveLength = int((1 - math.sqrt(area_perc)) * 300 / 2)

# big_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    
# 将模型加载到指定设备DEVICE上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = HRnet(num_classes=1,
                  backbone='hrnetv2_w32', pretrained=False, mask_thres = 0.9)
model.to(device)

big_image, geotransform, projection = readTif(file_path)
big_image_ndvi, _, _ = readTif(ndvi_file_path)

big_image = np.asarray(big_image)
big_image_ndvi = np.asarray(big_image_ndvi)
big_image = big_image / 255.0
big_image_ndvi = (big_image_ndvi + 1) / 2.0

big_image = torch.from_numpy(big_image)
big_image_ndvi = torch.from_numpy(big_image_ndvi)
big_image_ndvi = big_image_ndvi.unsqueeze(0)
big_image = torch.cat((big_image, big_image_ndvi), dim=0)

big_image = big_image.swapaxes(1, 0).swapaxes(1, 2)
TifArray, RowOver, ColumnOver = TifCroppingArray(big_image, RepetitiveLength)

for model_path in model_paths:
    model.eval()
    # Load the model
    state_dict = torch.load(model_path, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    # 移除 "module." 前缀
    new_state_dict = {k.replace('module.', ''): v for k,
                      v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    
model.n_classes =1
predicts_gr = []
predicts_building = []
for i in range(len(TifArray)):
    for j in range(len(TifArray[0])):
        image = TifArray[i][j]
        img = image.swapaxes(1, 2).swapaxes(1, 0)
        img = img.unsqueeze(0)
        img = img.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            output_gr, output_building = model(img)
            # output = F.interpolate(
            #     output, (image.shape[1], image.shape[0]), mode='bilinear')
            if model.n_classes > 1:
                predicted_mask_gr = output_gr.argmax(dim=1)
                predicted_mask_building = output_building.argmax(dim=1)                
            else:
                predicted_mask_gr = output_gr > 0.9
                predicted_mask_building = output_building > 0.9


        predicted_mask_gr = predicted_mask_gr[0].long().squeeze().cpu().numpy()
        predicted_mask_building = predicted_mask_building[0].long().squeeze().cpu().numpy()
        predicts_gr.append(predicted_mask_gr)
        predicts_building.append(predicted_mask_building)

# 保存结果predicts
result_shape = (big_image.shape[0], big_image.shape[1])
result_gr_data = Result(result_shape, TifArray, predicts_gr,
                     RepetitiveLength, RowOver, ColumnOver)
writeTiff(Result_GR_Path, result_gr_data, geotransform, projection)
result_building_data = Result(result_shape, TifArray, predicts_building,
                     RepetitiveLength, RowOver, ColumnOver)
writeTiff(Result_Building_Path, result_building_data, geotransform, projection)