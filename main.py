from src.loadDataset import load_data
from src.preprocessing import preprocessing_data
from src.model import train, malaria_cnn
import shutil
import os

if __name__ == '__main__':

    # Base directory
    base_dir_path = 'inputs/cell_images'
    target_size = (130, 130)
    # Download the data
    if 'cell_images' not in os.listdir('inputs'):
        load_data()
    if 'cell_images' in os.listdir(base_dir_path):
        shutil.rmtree(f'{base_dir_path}/cell_images')

    train_gen, validation_gen = preprocessing_data(base_dir_path, target_size=target_size)
    model = malaria_cnn(lr=0.001)
    history = train(model, train_gen, validation_gen, 15, 100, 50)

