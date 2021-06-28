from PIL import Image
from tqdm import tqdm
from colorama import Fore
import shutil
import numpy as np


def image_size(img):
    with Image.open(img) as _img:
        return _img.size


def move_files(files, original_path, destination_path, split=1):
    for file in tqdm(
            np.random.choice(
                files,
                round(len(files) * split),
                replace=False),
            desc=f'{Fore.YELLOW}Moving training files to: {destination_path}'):
        try:
            shutil.move(f'{original_path}/{file}', f'{destination_path}/{file}')
        except shutil.Error as e:
            print(e)
