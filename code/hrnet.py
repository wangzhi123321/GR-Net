import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hrnet import *


class HRnet_Backbone(nn.Module):
    def __init__(self, backbone='hrnetv2_w32', pretrained=False):
        super(HRnet_Backbone, self).__init__()
        self.model = hrnet_classification(
            backbone=backbone, pretrained=pretrained)
        del self.model.incre_modules
        del self.model.downsamp_modules
        del self.model.final_layer
        del self.model.classifier

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.conv2(x)
        x = self.model.bn2(x)
        x = self.model.relu(x)
        x = self.model.layer1(x)
        x = self.model.cbam_layers[3](x)

        x_list = []
        for i in range(2):
            if self.model.transition1[i] is not None:
                x_list.append(self.model.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.model.stage2(x_list)
        # 融入 CBAM 模块
        for i in range(len(y_list)):
            y_list[i] = self.model.cbam_layers[i](y_list[i])

        x_list = []
        for i in range(3):
            if self.model.transition2[i] is not None:
                if i < 2:
                    x_list.append(self.model.transition2[i](y_list[i]))
                else:
                    x_list.append(self.model.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.model.stage3(x_list)
        for i in range(len(y_list)):
            y_list[i] = self.model.cbam_layers[i](y_list[i])

        x_list = []
        for i in range(4):
            if self.model.transition3[i] is not None:
                if i < 3:
                    x_list.append(self.model.transition3[i](y_list[i]))
                else:
                    x_list.append(self.model.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.model.stage4(x_list)
        for i in range(len(y_list)):
            y_list[i] = self.model.cbam_layers[i](y_list[i])
        return y_list


class HRnet(nn.Module):
    def __init__(self,  num_classes=1, backbone='hrnetv2_w32', pretrained=False, mask_thres = 0.5):
        super(HRnet, self).__init__()
        self.backbone = HRnet_Backbone(
            backbone=backbone, pretrained=pretrained)

        last_inp_channels = np.int64(
            np.sum(self.backbone.model.pre_stage_channels))
        
        # 为NIR和NDVI通道定义一个单独的处理层
        self.additional_channels_conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=last_inp_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

        self.last_layer_out_building = nn.Sequential(
            nn.Conv2d(in_channels=last_inp_channels,
                      out_channels=last_inp_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.2),
            nn.Conv2d(in_channels=last_inp_channels,
                      out_channels=num_classes, kernel_size=1, stride=1, padding=0)

        )

        self.last_layer_out = nn.Sequential(
            nn.Conv2d(in_channels=last_inp_channels * 2,
                      out_channels=last_inp_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.2),
            nn.Conv2d(in_channels=last_inp_channels,
                      out_channels=num_classes, kernel_size=1, stride=1, padding=0)

        )
        

        self.cbam = CBAM(last_inp_channels * 2)
        self.mask_thres = mask_thres

    def filter(self, x_gr, x_building, mask_thres=0.5):
        correction = (x_building >= mask_thres).float()  
        x_gr = x_gr * correction
        return x_gr
    
    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        # 分割输入的bgr和NIR/NDVI通道
        inputs_bgr = inputs[:, :3, :, :]
        inputs_rgb = inputs_bgr[:, [2, 1, 0], :, :]
        inputs_additional = inputs[:, 3:, :, :]

        x = self.backbone(inputs_rgb)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w),
                           mode='bilinear', align_corners=True)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w),
                           mode='bilinear', align_corners=True)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w),
                           mode='bilinear', align_corners=True)

        x = torch.cat([x[0], x1, x2, x3], 1)

        x_building = self.last_layer_out_building(x)
        x_building = F.interpolate(x_building, size=(H, W), mode='bilinear', align_corners=True)
        x_building = x_building.sigmoid()


        # 单独处理NIR和NDVI通道
        x_additional = self.additional_channels_conv(inputs_additional)
        x_additional = F.interpolate(x_additional, size=(
            x0_h, x0_w), mode='bilinear', align_corners=True)
        
        x = torch.cat([x, x_additional], 1)
        x_gr = self.cbam(x)
        x_gr = self.last_layer_out(x_gr)
        x_gr = F.interpolate(x_gr, size=(H, W), mode='bilinear', align_corners=True)
        x_gr = x_gr.sigmoid()
        x_gr = self.filter(x_gr, x_building, self.mask_thres)

        return x_gr, x_building
    
