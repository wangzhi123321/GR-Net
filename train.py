import subprocess
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse  # 导入argparse模块，argparse可以帮助开发者编写易用且功能强大的命令行界面。
import logging  # 导入logging模块，该模块提供了可配置的日志记录功能
import os  # 导入os模块，os模块提供了一些与操作系统交互的函数
import random  # 导入Python的内置随机数生成器模块
import sys  # 导入sys模块，sys模块提供了与Python解释器和系统交互的函数和变量
import torch  # 入PyTorch深度学习框架的语句。
import torch.nn as nn  # 导入PyTorch深度学习库中的nn模块，其中包含了神经网络相关的类和函数。
import torch.nn.functional as F  # 导入torch库中的nn模块中的functional子模块，并将其命名为F。
# 导入了名为torchvision.transforms的Python模块，该模块提供了一系列用于图像增强和转换的函数和类。
import torchvision.transforms as transforms
# 导入了PyTorch的torchvision.transforms.functional模块，并将其别名为TF。该模块包含了一些对图像进行变换和处理的函数
import torchvision.transforms.functional as TF
from pathlib import Path  # 导入Path类，这个类可以用来表示文件路径和文件系统操作。
from torch import optim  # 导入了 PyTorch 深度学习框架中的优化器模块
# 从PyTorch库中导入两个模块：DataLoader（加载数据集）和random_split（将一个数据集随机分成两个子集）
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm  # 导入tqdm库中的tqdm函数
import numpy as np
import scipy


# 导入两个自定义数据集类BasicDataset和CarvanaDataset
from code.hrnet import HRnet
from code.utils.data_loading import BasicDataset, CarvanaDataset
# 从一个名为"utils"的Python模块中导入了名为"dice_score"的函数
from code.utils.dice_score import dice_loss
from code.utils.focal_loss import FocalBCELoss

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# 读取当前工作目录下的data/imgs/文件夹中的图像数据。使用"./"表示当前目录。
dir_img = Path('./data/train/imgs_bgrn_train_10000/')
dir_ndvi_img = Path('./data/train/ndvi_train_10000/')
dir_mask = Path('./data/train/masks_gr_train_10000/')  # 读取当前工作目录下的"data/masks/"文件夹中的掩码文件
dir_building_mask = Path('./data/train/masks_building_train_10000/') 

# 读取当前工作目录下的"checkpoints/"文件夹中的checkpoint文件
dir_checkpoint = Path(
    './runs/final')

if not os.path.exists(dir_checkpoint):
    os.makedirs(dir_checkpoint, exist_ok=True)


def print_epoch_train_info(cur_epoch, epochs, loss, epoch_start_time, epoch_end_time):

    print_msg = (
        f"[{cur_epoch}/{epochs}] "
        + f"loss: {loss:.9f} | "
        + "time:%.2f sec" % (epoch_end_time - epoch_start_time)
    )

    logging.info(print_msg)

    print(print_msg)


def init_logging(save_dir):
    # if not os.path.exists("log"):
    #     os.makedirs("log", exist_ok=True)

    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    filename = os.path.join(save_dir, "log.log")
    logging.basicConfig(
        filename=filename,
        format="%(asctime)s - %(pathname)s[line:%(lineno)d]: %(message)s",
        level=logging.INFO,
    )

    return filename

# 定义一个名为 train_model 的函数，它有多个参数。该函数的作用是训练机器学习模型，并返回训练好的模型。


def train_model(
        model,  # 表示要训练的模型
        device,  # 表示模型要在哪个设备（如 CPU 或 GPU）上运行
        world_size,
        epochs: int = 500,  # 表示训练周期数
        batch_size: int = 16,  # 表示每次迭代所使用的样本数
        learning_rate: float = 1e-3,  # 表示学习率
        val_percent: float = 20,  # 表示将数据集中的多少比例用于验证集
        save_checkpoint: bool = True,  # 表示是否保存模型检查点
        img_scale: float = 1,  # 表示图像缩放比例
        amp: bool = False,  # 表示是否启用自动混合精度
        weight_decay: float = 1e-8,  # 表示权重衰减系数
        momentum: float = 0.999,  # 表示动量参数
        gradient_clipping: float = 1.0,  # 表示梯度裁剪值
):
    # 加载数据
    # 1. Create dataset
    try:
        # 首先尝试创建一个名为CarvanaDataset的数据集对象，该对象需要三个参数：图像目录路径，掩膜目录路径和图像缩放比例。
        dataset = CarvanaDataset(dir_img, dir_mask, dir_building_mask, dir_ndvi_img, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        # 如果创建CarvanaDataset时出现AssertionError、RuntimeError或IndexError异常，则捕获这些异常并创建一个名为BasicDataset的基本数据集对象，该对象也需要相同的三个参数：图像目录路径，掩膜目录路径和图像缩放比例。换句话说，如果创建CarvanaDataset失败，则会回退到更基本的BasicDataset。
        dataset = BasicDataset(dir_img, dir_mask, dir_building_mask, dir_ndvi_img, img_scale)

    # 2. Create data loaders
    loader_args = dict(batch_size=batch_size//world_size,
                       num_workers=os.cpu_count(), pin_memory=False)  # 创建了一个包含三个参数的字典loader_args。第一个参数是batch_size，指定每个批次中包含的样本数。第二个参数是num_workers，这个参数设置为操作系统可用的CPU核心数量，即使用多线程进行数据加载以提高效率。最后一个参数是pin_memory，当该参数为True时，会将数据存储在固定的内存区域，加速GPU读取数据的速度。
    # 使用DataLoader函数构建了两个数据加载器：train_loader和val_loader。train_set和val_set是训练集和验证集的数据集对象。shuffle=True表示是否对数据进行随机排序，而drop_last=True表明如果最后一个批次不足一个完整的batch_size，则应该将其丢弃。最后，使用**loader_args将先前定义的字典参数传递给数据加载器。

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, shuffle=True)
    # val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, shuffle=False)

    train_loader = DataLoader(dataset,
                              sampler=train_sampler,
                              drop_last=True,
                              **loader_args)
    # val_loader = DataLoader(val_sampler, shuffle=False, drop_last=False, **loader_args)


    # 使用Python的logging模块来记录一些训练超参数和设置信息。其中，使用了f-string格式化字符串的语法，将变量值插入到输出的字符串中。
    if dist.get_rank() == 0:
        logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Training size:   {len(train_loader)}
            Checkpoints:     {save_checkpoint}
            Device:          {device.type}
            Images scaling:  {img_scale}
            Mixed Precision: {amp}
        ''')

    # 定义优化器、学习率的方案、损失函数
    # lr = args.lr * min(1, batch_size // 8)
    lr = args.lr
    # 3. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam优化器
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # 学习率衰减策略
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs, eta_min=5e-6)
    # optimizer = optim.RMSprop(model.parameters(),
    #                           lr=learning_rate, weight_decay=weight_decay, momentum=momentum) #创建了一个用于优化模型权重的RMSprop优化器，该优化器使用了学习率、权重衰减和动量等参数来控制优化过程。
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 'max', patience=50)  # goal: maximize Dice score.创建了一个根据验证集表现自适应调整学习率的调度器。
    # 创建了一个梯度缩放器，用于处理半精度浮点数数据（如果开启了混合精度训练）。这段代码使用了 PyTorch 中的 Automatic Mixed Precision (AMP) 技术，它可以自动地将深度学习模型中的计算转换为半精度浮点数格式，以提高训练速度和减少内存消耗。在这段代码中，grad_scaler 是一个用于缩放梯度的对象，它可以对反向传播过程中计算出的梯度进行缩放处理，以防止因使用半精度浮点数格式而导致的梯度下降不稳定或溢出等问题。enabled 参数指示是否启用 AMP 技术，如果设置为 True，则会启用 AMP，否则将不会使用。
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # criterion = FocalBCELoss() if args.classes > 1 else nn.BCEWithLogitsLoss() #根据模型输出的类别数选择相应的损失函数（交叉熵损失函数或二元交叉熵损失函数）。
    criterion = FocalBCELoss()
    global_step = 0  # 初始化全局步数变量为0。
    best_val_score = float('-inf')  # 初始化为正无穷大，确保第一个模型一定会保存
    t_loss = []  # 用于存放train_loss
    gr_loss = []
    building_loss = []
    v_score = []  # 用于存放val_score
# 开始训练
# epoch:对dataset做1轮完整训练。epochs表示总共需要训练的轮数
# batch: 一个批次——把数据集1分成多个批次。批次可以理解为一组具有相同属性的数据集合。
    # 4. Begin training
    for epoch in range(1, epochs + 1):  # 用来控制模型训练的循环次数
        import time
        epoch_start_time = time.time()
        model.train()  # 用于启动模型的训练过程
        train_loader.sampler.set_epoch(epoch)
        epoch_loss = 0  # 定义了一个变量 epoch_loss 并将其初始化为0。该变量可以用来记录该epoch中模型的总损失。
        epoch_loss1 = 0
        epoch_loss2 = 0

        # 使用了Python中的tqdm库，它可以用于在循环中实现进度条的显示
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:  # train_loader是一个数据加载器对象，用于将数据分成小批次进行训练。在每个循环迭代中，batch会被赋值为train_loader中的一个小批次数据，然后使用该批次数据来更新模型参数。
                # 将批次(batch)中的图像(images)和对应的真实掩码(true_masks)赋值给变量images和true_masks。
                images, true_masks, true_masks_building = batch['image'], batch['mask'], batch['building_mask']

                # # 断言语句，用于检测神经网络输入的图像通道数是否与预期一致。其含义为：如果输入的图像数据(images)中每个样本的通道数(images.shape[1])不等于神经网络模型(model)定义时指定的输入通道数(model.n_channels)，那么抛出一个 AssertionError 异常，同时输出一条错误信息。这个错误信息包括了当前模型所期望的输入通道数以及加载的图像数据实际上具有的通道数，请检查图像是否正确加载。
                # assert images.shape[1] == model.n_channels, \
                #     f'Network has been defined with {model.n_channels} input channels, ' \
                #     f'but loaded images have {images.shape[1]} channels. Please check that ' \
                #     'the images are loaded correctly.'

                images = images.to(
                    device=device, dtype=torch.float64, memory_format=torch.channels_last)  # 将一个名为"images"的张量转换到指定的设备上，并将其数据类型设置为32位浮点型。此外，还使用了内存格式参数(torch.channels_last)，它表示张量在内存中的布局方式，这里表示通道维度被放置在最后一个轴上，以加速计算
                # 将变量 true_masks 转换为在指定设备上的 torch.long 数据类型。
                true_masks = true_masks.to(device=device, dtype=torch.long)
                true_masks_building = true_masks_building.to(device=device, dtype=torch.long)

                # 前向传播
                # 使用 PyTorch 中的自动混合精度（Automatic Mixed Precision，简称 AMP）技术来加速模型训练，并减少显存占用。它的作用是将部分计算转换为低精度浮点数进行计算，以减少计算量和显存占用，同时又能保持模型的精度。
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    # 用了一个机器学习模型（model）来对一组图片（images）进行处理，得到了一组预测结果（masks_pred）
                    masks_pred, mask_building_pred = model(images)
                    # 损失函数计算部分
                    if args.classes == 1:  # 判断该模型是否为单类别分类任务
                        cls_loss1 = criterion(masks_pred.squeeze(1),
                                              true_masks.double())  # 使用交叉熵损失函数（criterion）计算预测结果与真实标签（true_masks）之间的误差。由于模型只有一个输出通道，需要对预测结果使用squeeze(1)函数去除通道维度，使得其与真实标签形状相同；同时使用float()将真实标签转换为浮点类型，以便与预测结果计算损失。
                        seg_loss1 = dice_loss(masks_pred.squeeze(1),
                                             true_masks.double(), multiclass=False)  # 使用Dice Loss计算相同张量之间的相似性。这里使用了F.sigmoid(masks_pred.squeeze(1))函数将预测结果映射到0-1之间的概率值，并将multiclass参数设置为False表示仅针对单一类别计算Dice Loss。最后，将两个损失值相加得到总的损失函数值。
                        loss1 = cls_loss1 + seg_loss1 

                        cls_loss2 = criterion(mask_building_pred.squeeze(1),
                                              true_masks_building.double())  # 使用交叉熵损失函数（criterion）计算预测结果与真实标签（true_masks）之间的误差。由于模型只有一个输出通道，需要对预测结果使用squeeze(1)函数去除通道维度，使得其与真实标签形状相同；同时使用float()将真实标签转换为浮点类型，以便与预测结果计算损失。
                        
                        seg_loss2 = dice_loss(mask_building_pred.squeeze(1),
                                             true_masks_building.double(), multiclass=False)  # 使用Dice Loss计算相同张量之间的相似性。这里使用了F.sigmoid(masks_pred.squeeze(1))函数将预测结果映射到0-1之间的概率值，并将multiclass参数设置为False表示仅针对单一类别计算Dice Loss。最后，将两个损失值相加得到总的损失函数值。
                        
                        loss2 = cls_loss2 + seg_loss2
                        loss = loss1 + loss2
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).double(),
                            F.one_hot(true_masks, args.classes).permute(
                                0, 3, 1, 2).double(),
                            multiclass=True
                        )  # 如果两者的维度不一致，则需要将预测掩膜经过softmax函数转换为概率分布，并使用F.one_hot函数将真实掩膜转化为独热编码格式，以便进行Dice Loss的计算。最后将Dice Loss和交叉熵损失相加，得到总损失。其中，multiclass=True表示处理多类别的情况。

                # 反向传播
                # 梯度置零，初始化.通过调用 optimizer 对象的 zero_grad() 方法来清空梯度信息。set_to_none=True 的参数设置表示将梯度张量置为 None，这样可以释放内存并减少计算负担。这个方法通常在每次进行反向传播之前调用，以避免梯度累积导致计算错误。
                optimizer.zero_grad(set_to_none=True)
                # 反向传播. 使用了 PyTorch 中的自动求导功能来计算神经网络的梯度，并且还使用了一个梯度缩放器（GradientScaler）来控制梯度值的大小。具体来说，grad_scaler.scale(loss) 将损失值 loss 乘以梯度缩放器的比例因子，得到缩放后的损失值，并将其作为反向传播的起点。在反向传播的过程中，根据链式法则，每个参数节点都会计算出其对应的梯度值，最终形成完整的梯度信息。最后，这些梯度值可以用于更新神经网络的参数，从而优化模型的性能。
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clipping)  # 使用 PyTorch 的 nn.utils 模块中的 clip_grad_norm_ 函数对模型参数的梯度进行裁剪
                grad_scaler.step(optimizer)  # 参数更新
                grad_scaler.update()  # 更新梯度缩放因子

                # 使用一个名为 pbar 的进度条对象来更新当前处理的图像数量。pbar.update() 方法用于将已完成的进度加入到进度条中，参数 images.shape[0] 返回 images 数组的第一维大小，也就是当前正在处理的图像数量。
                pbar.update(images.shape[0])
                global_step += 1  # 一个计数器，它会在每次被调用时将全局变量 "global_step" 的值增加 1。这个计数器通常用于在训练神经网络等机器学习模型时跟踪迭代次数。
                epoch_loss += loss.item()  # 计算一个 epoch（一个完整的数据集被模型遍历一次）中所有 batch 的损失函数值的总和。
                epoch_loss1 += loss1.item()
                epoch_loss2 += loss2.item()
                # experiment.log({
                #     'train loss': loss.item(),
                #     'step': global_step,
                #     'epoch': epoch
                # })
                if dist.get_rank() == 0:
                    # 在进度条上方显示当前批次的损失值。
                    pbar.set_postfix(**{'loss (batch)': loss.item()})

            train_loss = epoch_loss/len(train_loader)
            losses1 = epoch_loss1/len(train_loader)
            losses2 = epoch_loss2/len(train_loader)

            t_loss.append(train_loss)
            gr_loss.append(losses1)
            building_loss.append(losses2)
            scheduler.step(train_loss)

            epoch_end_time = time.time()
            print_epoch_train_info(
                epoch, epochs, train_loss, epoch_start_time, epoch_end_time)


        if dist.get_rank() == 0 and save_checkpoint and epoch % 10 == 0:  # 每50轮保存一个检查点
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            checkpoint_path = dir_checkpoint / \
                'checkpoint_epoch{}.pth'.format(epoch)
            torch.save(state_dict, str(checkpoint_path))
            logging.info(f'Checkpoint {epoch} saved!')

    if dist.get_rank() == 0:
        with open("./loss/Ablation_study/train_loss_final_50_5.txt", 'w') as t_los:
            for value in t_loss:
                t_los.write(str(value) + "\n")
    if dist.get_rank() == 0:
        with open("./loss/Ablation_study/gr_loss_final_50_5.txt", 'w') as gr_los:
            for value in gr_loss:
                gr_los.write(str(value) + "\n")
    if dist.get_rank() == 0:
        with open("./loss/Ablation_study/building_loss_final_50_5.txt", 'w') as building_los:
            for value in building_loss:
                building_los.write(str(value) + "\n")

        # with open("./loss/val_score.txt", 'w') as v_scor:
        #         for value in v_score:
        #             v_scor.write(str(value) + "\n")
"""
定义参数
"""


def get_args():  # 定义一个名为“get_args”的函数
    import datetime  # 导入一个名为datetime的模块。datetime模块提供了处理日期和时间的函数和类。
    now = datetime.datetime.now()  # 使用Python中的datetime模块创建了一个对象now，该对象包含当前的日期和时间信息
    # 使用了strftime()函数对当前时间进行格式化，将年份、月份、日期、小时数、分钟数、秒数和毫秒数以一定的格式组合在一起，用下划线分隔开来，最终生成一个类似于"20230424_155230_123456"的时间戳字符串
    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
    parser = argparse.ArgumentParser(
        description='Train the UNet on images and target masks')  # 创建一个用于解析命令行参数的对象，名称为parser
    parser.add_argument('--timestamp',
                        type=str, default=timestamp, help='timestamp')  # 使用 Python 中的 argparse 模块来解析命令行参数。使用`parser.add_argument()` 方法向解析器中添加参数，参数名称为训练周期数，字符串类型，默认值为变量 timestamp 的当前值，帮助信息
    parser.add_argument('--epochs', '-e', metavar='E',
                        type=int, default=50, help='Number of epochs')  # 参数名称为轮次，--epochs在帮助信息中的显示格式为E，整型，默认值为5，
    parser.add_argument('--batch-size', '-b',
                        type=int, default=16, help='Batch size')  # 添加了一个名为'--batch-size'（或'-b'）的参数，它的类型为整数（type=int），默认值为1（default=1），并且还提供了一个帮助信息（help='Batch size'）。这个参数可以用来指定批次的大小。
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-3,
                        help='Learning rate', dest='lr')  # 学习率。添加了一个名为 "--learning-rate" 或 "-l" 的可选参数。该参数接受一个浮点数类型的值，并设置默认值为 1e-5。定义了一个元变量（metavar），即在帮助信息中显示的参数名称，这里为 "LR"。同时也定义了一个别名（dest）为 "lr"，当在代码中引用该参数时，可以使用别名 "lr" 来代替参数名 "--learning-rate" 或 "-l"。
    parser.add_argument('--load', '-f', type=str,
                        default='runs/bubbliiing_pretrained/hrnetv2_w32_weights_voc.pth', help='Load model from a .pth file')  # 模型加载路径。加命令行参数的过程中添加一个名为 "--load" 或 "-f" 的参数。这个参数是字符串类型，并且默认值为 False，用于指示是否从一个.pth文件中加载模型。
    parser.add_argument('--scale', '-s', type=float,
                        default=1, help='Downscaling factor of the images')  # 参数名为 "--scale" 和 "-s"，表示可用 "--scale" 或 "-s" 选项来调用该参数。它们的值是浮点数类型，缺省值为 0.5。help 参数用于描述该参数的用途，即图像的缩放因子(downscaling factor)。
    parser.add_argument('--validation', '-v', dest='val', type=float, default=0,
                        help='Percent of the data that is used as validation (0-100)')  # 加了一个名为"--validation"（或"-v"）的可选参数，并将其目标属性设置为"val"。此参数接受浮点值类型的输入，默认值为10.0，并提供了一个帮助文本，指示此参数代表数据集中用于验证的百分比（0-100）。
    parser.add_argument('--amp',
                        default=False, help='Use mixed precision')  # 义了一个名为 "--amp" 的命令行参数。该参数默认值为 False，表示不使用混合精度技术。该参数的作用是帮助提高训练速度和节省 GPU 内存，可以通过设置为 True 来启用混合精度技术。
    parser.add_argument('--bilinear', action='store_true',
                        default=False, help='Use bilinear upsampling')  # 定义了一个名为 "bilinear" 的命令行参数，它是一个布尔值（True 或 False），可以通过在命令行中输入 "--bilinear" 来设置其值为 True。当不提供该参数时，它的默认值为 False。表示是否进行双线性上采样
    parser.add_argument('--classes', '-c', type=int,
                        default=1, help='Number of classes')  # 添加一个名为 "--classes" 或 "-c" 的参数，其类型为整数（int），默认值为 2，
    parser.add_argument('--channels', type=int,
                        default=5, help='Number of channels')
    parser.add_argument("-random_seed", default=42,
                        type=int, help="random seed")
    return parser.parse_args()


def setup_DDP():
    rank = int(os.environ["SLURM_PROCID"])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    local_rank = rank % num_gpus
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = "39500"
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['RANK'] = str(rank)

    print(os.environ["MASTER_PORT"], os.environ["MASTER_ADDR"],
          os.environ["LOCAL_RANK"], os.environ["RANK"], os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    print("torch.cuda.device_count():", torch.cuda.device_count())

    print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")

    return rank, local_rank, ntasks, device


def init_seeds(random_seed):
    '''固定各类随机种子，方便消融实验.
    Args:
        seed :  int
    '''
    # 固定 scipy 的随机种子
    random.seed(random_seed)  # 固定 random 库的随机种子
    os.environ['PYTHONHASHSEED'] = str(random_seed)  # 固定 python hash 的随机性（并不一定有效）
    np.random.seed(random_seed)  # 固定 numpy  的随机种子
    torch.manual_seed(random_seed)  # 固定 torch cpu 计算的随机种子
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(random_seed)
    else:
        torch.cuda.manual_seed(random_seed) # 固定 cuda 计算的随机种子
    torch.backends.cudnn.deterministic = True  # 是否将卷积算子的计算实现固定。torch 的底层有不同的库来实现卷积算子
    torch.backends.cudnn.benchmark = False  # 是否开启自动优化，选择最快的卷积计算方法



if __name__ == '__main__':  # 运行脚本时从此处开始运行
    rank, local_rank, world_size, device = setup_DDP()
    # 参数定义
    args = get_args()  # 定义模型参数

   # 打印日志
    if dist.get_rank() == 0:
        init_logging(dir_checkpoint)
    # Settings
    init_seeds(args.random_seed)

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    # 定义模型
    # 创建了一个名为"model"的UNet模型，该模型有3个输入通道（即RGB图像），输出类别数由参数args.classes指定，bilinear参数指定是否使用双线性插值
    model = HRnet(num_classes=args.classes,
                  backbone='hrnetv2_w32', pretrained=False, mask_thres = 0.5)
    # model = UNet(n_channels=5, n_classes=args.classes, bilinear=args.bilinear)
    # 将 PyTorch 模型的内存格式(memory format)转换为通道优先(channels last)的格式
    model = model.to(memory_format=torch.channels_last)
    model = model.double()

    # 加载模型
    # 根据命令行参数args.load加载一个预训练模型的状态字典（state_dict），并将其加载到当前的PyTorch模型中（model）。其中，map_location参数指定了在哪个设备上加载模型（例如CPU或GPU）。读入的状态字典中包含了该预训练模型的所有参数，包括权重、偏置等。
    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict, strict=False)  # 把权重加载到model
        if dist.get_rank() == 0:
            # 通过日志输出显示模型已经成功地从指定路径(args.load)加载
            logging.info(f'Model loaded from {args.load}')

    model.to(device=device)  # 将PyTorch模型移动（或复制）到指定的设备上运行，其中device参数指定了目标设备。
    # Convert to DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
 
    train_model(
        model=model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        img_scale=args.scale,
        val_percent=args.val / 100,
        amp=args.amp,
        world_size=world_size,
    )

