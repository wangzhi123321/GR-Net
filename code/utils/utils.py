import os
import matplotlib.pyplot as plt
import numpy as np
import imageio

def plot_mask(mask, filename, save_dir, target_class, output_size):
    result = (mask == target_class).astype(np.uint8) * 255

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{filename.split('.')[0]}_pred.tif")
    imageio.imwrite(save_path, result)


