import argparse
import logging
import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from code.utils.data_loading import BasicDataset
# from unet import UNet  # 从名为"unet"的模块中导入了一个名为"UNet"的类
# from bubbliiiing.hrnet_cbam import HRnet
# from unet.unet16 import UNetBackbone
# from bubbliiiing.hrnet_cbam2 import HRnet
from code.bubbliiiing.hrnet_1.hrnet_cbam_out_building_rgb3 import HRnet
# from bubbliiiing.hrnet_cbam5 import HRnet
from code.utils.utils import plot_mask


# def predict_img(net,
#                 full_img,
#                 device,
#                 scale_factor=1,
#                 out_threshold=0.5):
#     net.eval()
#     img = torch.from_numpy(BasicDataset.preprocess(
#         None, full_img, scale_factor, is_mask=False))
#     img = img.unsqueeze(0)
#     img = img.to(device=device, dtype=torch.float32)

#     with torch.no_grad():
#         output = net(img).cpu()
#         output = F.interpolate(
#             output, (full_img.size[1], full_img.size[0]), mode='bilinear')
#         if net.n_classes > 1:
#             mask = output.argmax(dim=1)
#         else:
#             mask = torch.sigmoid(output) > out_threshold

#     return mask[0].long().squeeze().numpy()

def get_args():
    parser = argparse.ArgumentParser( 
        description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='runs/Ablation_study/final_epoch50/checkpoint_epoch50.pth', type=str,
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input_img', '-i', default='data/10000',
                        help='Filenames of input images')
    parser.add_argument('--input_ndvi_img', '-in', default='data/10000_ndvi',
                        help='Filenames of input ndvi images')
    parser.add_argument('--output', '-o', metavar='OUTPUT',
                        nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', default=True, type=bool,
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', default=True, type=bool,
                        help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.1,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true',
                        default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int,
                        default=1, help='Number of classes')
    parser.add_argument('--output_green_roof', '-og', type=str, default="data/10000_greenroof",
                    help='Directory to save output images for green roofs')
    parser.add_argument('--output_building', '-ob', type=str, default="data/10000_building",
                    help='Directory to save output images for buildings')
    parser.add_argument('--gt_dir', type=str,
                        default="data/10000_mask", help='data/10000_mask')
    parser.add_argument('--gt_building_dir', type=str,
                        default="data/10000_building_mask", help='pre_data/mask_building_test')
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn, suffix):
        # 添加后缀以区分两类输出
        return f'{os.path.splitext(fn)[0]}_{suffix}.png'

    input_files = os.listdir(args.input_img)
    if not args.output:
        # 创建两组文件名，一组用于绿色屋顶，另一组用于建筑
        green_roof_files = list(map(lambda fn: _generate_name(fn, "green_roof"), input_files))
        building_files = list(map(lambda fn: _generate_name(fn, "building"), input_files))
        return green_roof_files, building_files
    else:
        # 如果args.output已经定义了输出文件名，直接返回这些文件名的两个副本（通常不建议这样使用，需根据实际情况调整）
        return args.output, args.output


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1],
                       len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


def calculate_pixel_accuracy(predicted, target):
    # pred是模型的预测结果，target是真实标签
    # pred和target都是二维数组，每个元素代表一个像素的类别

    # 将预测结果和真实标签转换为一维数组
    predicted_flat = predicted.flatten()
    target_flat = target.flatten()
    # 计算被正确分类的像素数量
    correct_pixels = np.sum(predicted_flat == target_flat)
    # 计算总像素数量
    total_pixels = len(target_flat)
    # 计算像素准确率
    pixel_accuracy = correct_pixels / total_pixels

    return pixel_accuracy


def calculate_dice_score(predicted, target):
    intersection = np.logical_and(predicted, target).sum()
    union = np.logical_or(predicted, target).sum()
    dice_score = (2.0 * intersection) / (union + intersection)
    return dice_score


def calculate_iou(predicted, target):
    intersection = np.logical_and(predicted, target).sum()
    union = np.logical_or(predicted, target).sum()
    iou = intersection / union if union > 0 else 0.0
    return iou


def calculate_precision_recall(predicted, target):
    true_positive = np.logical_and(predicted == 1, target == 1).sum()
    false_positive = np.logical_and(predicted == 1, target == 0).sum()  # 误报
    true_negative = np.logical_and(predicted == 0, target == 0).sum()
    false_negative = np.logical_and(predicted == 0, target == 1).sum()  # 漏报

    precision = true_positive / \
        (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / \
        (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0

    return true_positive, false_positive, true_negative, false_negative, precision, recall


def calculate_f1_score(precision, recall):
    f1_score = 2 * (precision * recall) / (precision +
                                           recall) if (precision + recall) > 0 else 0.0
    return f1_score


def predict_and_evaluate(net, full_img, ndvi_img, gt_mask, gt_building_mask, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    args = get_args()
    # Preprocess the original image
    img = torch.from_numpy(BasicDataset.preprocess(
        None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    # Preprocess the NDVI image
    ndvi = torch.from_numpy(BasicDataset.preprocess(
        None, ndvi_img, scale_factor, is_mask=False))
    ndvi = ndvi.unsqueeze(0)
    ndvi = ndvi.to(device=device, dtype=torch.float32)
    img = torch.cat((img, ndvi), dim=1)

    with torch.no_grad():
        # output = net(img).cpu()
        output_gr, output_building = net(img)
        # output_gr = F.interpolate(
        #     output_gr, (full_img.shape[1], full_img.shape[0]), mode='bilinear')
        if args.classes > 1:
            predicted_mask = output_gr.argmax(dim=1)
            predicted_building_mask = output_building.argmax(dim=1)            
        else:
            predicted_mask = output_gr > out_threshold
            predicted_building_mask = output_building > out_threshold        

    predicted_mask = predicted_mask[0].long().squeeze().cpu().numpy()
    predicted_building_mask = predicted_building_mask[0].long().squeeze().cpu().numpy()

    # Convert ground truth mask to numpy array
    gt_mask = np.array(gt_mask)
    gt_building_mask = np.array(gt_building_mask)

    pixel_accuracy_gr = calculate_pixel_accuracy(predicted_mask, gt_mask)
    # Calculate Dice Score
    dice_score_gr = calculate_dice_score(predicted_mask, gt_mask)
    iou_gr = calculate_iou(predicted_mask, gt_mask)
    # 计算 Precision、Recall 和 F1 分数
    true_positive_gr, false_positive_gr, true_negative_gr, false_negative_gr, precision_gr, recall_gr = calculate_precision_recall(
        predicted_mask, gt_mask)
    # f1_score = calculate_f1_score(precision, recall)
    
    pixel_accuracy_building = calculate_pixel_accuracy(predicted_building_mask, gt_building_mask)
    # Calculate Dice Score
    dice_score_building = calculate_dice_score(predicted_building_mask, gt_building_mask)
    iou_building = calculate_iou(predicted_building_mask, gt_building_mask)
    # 计算 Precision、Recall 和 F1 分数
    true_positive_building, false_positive_building, true_negative_building, false_negative_building, precision_building, recall_building = calculate_precision_recall(
        predicted_building_mask, gt_building_mask)
    # f1_score = calculate_f1_score(precision, recall)
    return predicted_mask, pixel_accuracy_gr, dice_score_gr, iou_gr, true_positive_gr, false_positive_gr, true_negative_gr, false_negative_gr, \
           predicted_building_mask, pixel_accuracy_building, dice_score_building, iou_building, true_positive_building, false_positive_building, true_negative_building, false_negative_building


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')

    in_files = os.listdir(args.input_img)
    # out_files = get_output_filenames(args)
    green_roof_files, building_files = get_output_filenames(args)

    net = HRnet(num_classes=args.classes,
                  backbone='hrnetv2_w32', pretrained=False, mask_thres = 0.1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    # if args.model:
    #     # Load the model
    #     state_dict = torch.load(args.model, map_location=device)
    #     mask_values = state_dict.pop('mask_values', [0, 1])
    #     # 移除 "module." 前缀
    #     new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    #     net.load_state_dict(new_state_dict)
    # else:   
    #     # Download the pretrained model
    #     print("Loading pretrained model...")
    #     net = torch.hub.load('milesial/Pytorch-UNet',
    #                          'unet_carvana', pretrained=True, scale=args.scale)

    if args.model:
        # Load the model
        state_dict = torch.load(args.model, map_location=device)
        mask_values = state_dict.pop('mask_values', [0, 1])
        # # 移除 "module." 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            # 删除第一个 'module.' 前缀
            new_key = re.sub(r'^module\.', '', k, count=1)
            new_key = re.sub(r'^module\.', '', new_key, count=1)
            new_state_dict[new_key] = v

        net.load_state_dict(new_state_dict)
    else:
        # Download the pretrained model
        print("Loading pretrained model...")
        net = torch.hub.load('milesial/Pytorch-UNet',
                             'unet_carvana', pretrained=True, scale=args.scale)

    logging.info('Model loaded!')
    net.to(device=device)

    total_pixel_accuracy_gr = 0.0
    total_dice_score_gr = 0.0
    total_iou_gr = 0.0
    # total_f1_score = 0.0
    total_true_positive_gr = 0.0
    total_false_positive_gr = 0.0
    total_true_negative_gr = 0.0
    total_false_negative_gr = 0.0

    total_pixel_accuracy_building = 0.0
    total_dice_score_building = 0.0
    total_iou_building = 0.0
    # total_f1_score_building = 0.0
    total_true_positive_building = 0.0
    total_false_positive_building = 0.0
    total_true_negative_building = 0.0
    total_false_negative_building = 0.0

    for i, filename in enumerate(in_files):
        file_path = os.path.join(args.input_img, filename)
        ndvi_file_path = os.path.join(
            args.input_ndvi_img, filename.replace('.tif', '_ndvi.tif'))  # 调整命名规则

        logging.info(f'Predicting and evaluating image {file_path} ...')
        img = Image.open(file_path)
        ndvi_img = Image.open(ndvi_file_path)
        img = np.asarray(img)
        ndvi_img = np.asarray(ndvi_img)
        img = img / 255.0
        ndvi_img = (ndvi_img + 1) / 2.0
        # Load the ground truth mask
        ground_truth_mask_path = os.path.join(
            args.gt_dir, filename.replace('.tif', '_mask.tif'))
        ground_truth_building_mask_path = os.path.join(args.gt_building_dir, filename.replace('.tif', '_building_mask.tif'))

        ground_truth_mask = Image.open(ground_truth_mask_path)
        ground_truth_building_mask = Image.open(ground_truth_building_mask_path)

        # Predict and evaluate
        predicted_mask, pixel_accuracy_gr, dice_score_gr, iou_gr, true_positive_gr, false_positive_gr, true_negative_gr, false_negative_gr, \
        predicted_building_mask, pixel_accuracy_building, dice_score_building, iou_building, true_positive_building, false_positive_building, true_negative_building, false_negative_building  = predict_and_evaluate(
            net, img, ndvi_img, ground_truth_mask, ground_truth_building_mask, device, args.scale, args.mask_threshold)

        total_pixel_accuracy_gr += pixel_accuracy_gr
        total_dice_score_gr += dice_score_gr
        total_iou_gr += iou_gr
        # total_f1_score += f1_score
        total_true_positive_gr += true_positive_gr
        total_false_positive_gr += false_positive_gr
        total_true_negative_gr += true_negative_gr
        total_false_negative_gr += false_negative_gr

        total_pixel_accuracy_building += pixel_accuracy_building
        total_dice_score_building += dice_score_building
        total_iou_building += iou_building
        # total_f1_score += f1_score
        total_true_positive_building += true_positive_building
        total_false_positive_building += false_positive_building
        total_true_negative_building += true_negative_building
        total_false_negative_building += false_negative_building

        if not args.no_save:
            # 生成绿色屋顶的输出文件名和路径
            out_filename_green_roof = os.path.join(args.output_green_roof, green_roof_files[i])
            result_green_roof = mask_to_image(predicted_mask, mask_values)
            result_green_roof.save(out_filename_green_roof)
            logging.info(f'Green roof mask saved to {out_filename_green_roof}')

            # 生成建筑的输出文件名和路径
            out_filename_building = os.path.join(args.output_building, building_files[i])
            result_building = mask_to_image(predicted_building_mask, mask_values)
            result_building.save(out_filename_building)
            logging.info(f'Building mask saved to {out_filename_building}')

        if args.viz:
            logging.info(f'Visualizing results for green roof in image {file_path}...')
            plot_mask(predicted_mask, filename, args.output_green_roof, target_class=1, output_size=(300, 300))
            
            logging.info(f'Visualizing results for building in image {file_path}...')
            plot_mask(predicted_building_mask, filename, args.output_building, target_class=1, output_size=(300, 300))


    average_pixel_accuracy_gr = total_pixel_accuracy_gr / len(in_files)
    average_dice_score_gr = total_dice_score_gr / len(in_files)
    average_iou_gr = total_iou_gr / len(in_files)
    # average_f1_score = total_f1_score / len(in_files)
    average_true_positive_gr = total_true_positive_gr / len(in_files)
    average_false_positive_gr = total_false_positive_gr / len(in_files)
    average_true_negative_gr = total_true_negative_gr / len(in_files)
    average_false_negative_gr = total_false_negative_gr / len(in_files)

    average_pixel_accuracy_building = total_pixel_accuracy_building / len(in_files)
    average_dice_score_building = total_dice_score_building / len(in_files)
    average_iou_building = total_iou_building / len(in_files)
    # average_f1_score = total_f1_score / len(in_files)
    average_true_positive_building = total_true_positive_building / len(in_files)
    average_false_positive_building = total_false_positive_building / len(in_files)
    average_true_negative_building = total_true_negative_building / len(in_files)
    average_false_negative_building = total_false_negative_building / len(in_files)

    logging.info(f'Green Roof:')
    logging.info(f'pixel_accuracy: {average_pixel_accuracy_gr}')
    logging.info(f'Dice_score: {average_dice_score_gr}')
    logging.info(f'IoU: {average_iou_gr}')
    # logging.info(f'测试集上的平均 f1 分数: {average_f1_score}')
    logging.info(f'TP: {average_true_positive_gr}')
    logging.info(f'FP: {average_false_positive_gr}')
    logging.info(f'TN: {average_true_negative_gr}')
    logging.info(f'FN: {average_false_negative_gr}')

    logging.info(f'Building:')
    logging.info(f'pixel_accuracy: {average_pixel_accuracy_building}')
    logging.info(f'Dice_score: {average_dice_score_building}')
    logging.info(f'IoU: {average_iou_building}')
    # logging.info(f'测试集上的平均 f1 分数: {average_f1_score}')
    logging.info(f'TP: {average_true_positive_building}')
    logging.info(f'FP: {average_false_positive_building}')
    logging.info(f'TN: {average_true_negative_building}')
    logging.info(f'FN: {average_false_negative_building}')

if __name__ == '__main__':
    main()
