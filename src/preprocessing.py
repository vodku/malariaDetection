from .utils import move_files
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from colorama import Fore, init
import numpy as np
import os


def preprocessing_data(base_dir_path: str, split: float = 0.8, target_size: tuple = (130, 130)):

    # Set colorama autorest
    init(autoreset=True)
    # Setup the seed
    np.random.seed(0)

    # Creating the directories
    base_dirs = os.listdir(base_dir_path)
    train_dir = os.path.join(base_dir_path, 'train')
    validation_dir = os.path.join(base_dir_path, 'validation')
    train_dir_uninfected = os.path.join(train_dir, 'Uninfected')
    train_dir_parasites = os.path.join(train_dir, 'Parasitized')
    validation_dir_uninfected = os.path.join(validation_dir, 'Uninfected')
    validation_dir_parasites = os.path.join(validation_dir, 'Parasitized')

    # Split the data into training and validation
    split = split
    uninfected_path = os.path.join(base_dir_path, 'Uninfected')
    parasites_path = os.path.join(base_dir_path, 'Parasitized')
    uninfected = os.listdir(uninfected_path)
    parasites = os.listdir(parasites_path)

    # Create training and validation directory
    if 'train' not in base_dirs:
        os.mkdir(train_dir)
        for d in ['Uninfected', 'Parasitized']:
            os.mkdir(os.path.join(train_dir, d))
        # Move training and validation files
        move_files(uninfected, uninfected_path, train_dir_uninfected, split)
        move_files(parasites, parasites_path, train_dir_parasites, split)
    else:
        print(Fore.GREEN + 'Training directory already created.')

    if 'validation' not in base_dirs:
        uninfected = os.listdir(uninfected_path)
        parasites = os.listdir(parasites_path)
        os.mkdir(validation_dir)
        for d in ['Uninfected', 'Parasitized']:
            os.mkdir(os.path.join(validation_dir, d))
        # Move training and validation files
        move_files(uninfected, uninfected_path, validation_dir_uninfected)
        os.rmdir(uninfected_path)
        move_files(parasites, parasites_path, validation_dir_parasites)
        os.rmdir(parasites_path)
    else:
        print(Fore.GREEN + 'Validation directory already created.')

    # Rescale the images to 130x130
    train_datagen = ImageDataGenerator(rescale=1./255.)
    validation_datagen = ImageDataGenerator(rescale=1./255.)
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        batch_size=20,
                                                        class_mode='binary',
                                                        target_size=target_size)
    validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                                  batch_size=20,
                                                                  class_mode='binary',
                                                                  target_size=target_size)
    return train_generator, validation_generator
