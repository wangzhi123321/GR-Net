import logging
import numpy as np
import torch
from PIL import Image
import cv2
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision.transforms import ToPILImage

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)

def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')
    
def unique_building_mask_values(idx, building_mask_dir, building_mask_suffix):
    building_mask_file = list(building_mask_dir.glob(idx + building_mask_suffix + '.*'))[0]
    building_mask = np.asarray(load_image(building_mask_file))
    if building_mask.ndim == 2:
        return np.unique(building_mask)
    elif building_mask.ndim == 3:
        building_mask = building_mask.reshape(-1, building_mask.shape[-1])
        return np.unique(building_mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {building_mask.ndim}')



class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, building_mask_dir: str, ndvi_dir: str, scale: float = 1.0, mask_suffix: str = '', building_mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.building_mask_dir = Path(building_mask_dir)
        self.ndvi_dir = Path(ndvi_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.building_mask_suffix = building_mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')

        with Pool() as p:
            unique_1 = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique_1), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

        with Pool() as p:
            unique_2 = list(tqdm(
                p.imap(partial(unique_building_mask_values, building_mask_dir=self.building_mask_dir, building_mask_suffix=self.building_mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.building_mask_values = list(sorted(np.unique(np.concatenate(unique_2), axis=0).tolist()))
        logging.info(f'Unique building mask values: {self.building_mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.shape[0], pil_img.shape[1]
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = cv2.resize(pil_img, (newW, newH), interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC)
        # pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            # if (img > 1).any():
            #     img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        building_mask_file = list(self.building_mask_dir.glob(name + self.building_mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(building_mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {building_mask_file}'        
        
        building_mask = load_image(building_mask_file[0])
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])
        assert img.size == mask.size == building_mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size}, {mask.size} and {building_mask.size}'

        # 加载NDVI图像
        ndvi_file = list(self.ndvi_dir.glob(name + '_ndvi.*'))  # 调整命名规则
        assert len(ndvi_file) == 1, f'ID为{name}的NDVI图像文件数不为1:{ndvi_file}'
        ndvi_img = load_image(ndvi_file[0])
        assert img.size == ndvi_img.size, f'RGBN图像和NDVI图像{name}应该具有相同的大小，但它们的大小分别为{img.size}和{ndvi_img.size}'
        # 将NDVI作为新通道连接
        ndvi_img = np.asarray(ndvi_img)  # 将ndvi_img转换为numpy数组

        # 将单通道的灰度图像扩展为一个通道，复制一次
        #ndvi_img = np.repeat(ndvi_img[:, :, np.newaxis], 1, axis=-1)
        ndvi_img = np.expand_dims(ndvi_img, axis=-1)
        mask = np.asarray(mask)
        building_mask = np.asarray(building_mask)
        img = np.asarray(img)
        img = img / 255.0
        ndvi_img= (ndvi_img + 1) / 2.0

        # 连接图像
        img = np.concatenate([img, ndvi_img], axis=-1)
        # img = np.stack([img, ndvi_img], axis=-1)
        # 将NumPy数组转换为图像
        # img = img.astype(np.uint8)
        # img = Image.fromarray(img)

        # 连接图像
        #img_channels = list(img.split())
        #img_channels.append(ndvi_img.convert('L'))  # 将NDVI添加到通道列表的末尾
        #img = Image.merge('RGBN', tuple(img_channels))

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)
        building_mask = self.preprocess(self.building_mask_values, building_mask, self.scale, is_mask=True)
        return {
            'image': torch.as_tensor(img.copy()).double().contiguous(),
            # 'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'building_mask': torch.as_tensor(building_mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, building_mask_dir, ndvi_dir, scale=1):
        super().__init__(images_dir, mask_dir, building_mask_dir, ndvi_dir, scale, mask_suffix='_mask', building_mask_suffix='_building_mask')
