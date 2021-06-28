import subprocess
import os
from zipfile import ZipFile, error
from colorama import Fore
from tqdm import tqdm


def load_data():

    filepath = 'inputs/cell-images-for-detecting-malaria.zip'
    output = 'inputs/'
    if 'cell-images-for-detecting-malaria.zip' not in os.listdir('inputs'):
        print(f'{Fore.YELLOW}Downloading the dataset from Kaggle...')
        subprocess.run(f'kaggle datasets download -d iarunava/cell-images-for-detecting-malaria -p {output}',
                       stdout=subprocess.PIPE,
                       shell=True)
        print(f'{Fore.YELLOW}Download done.')

    with ZipFile(filepath, 'r') as zipObj:
        try:
            for i in tqdm(zipObj.infolist(), desc=f'{Fore.YELLOW}Extracting:'):
                zipObj.extract(i, output)
            print(f'{Fore.GREEN}Files successfully extracted.')
            return True
        except error as e:
            print(f'{Fore.RED}Cannot extract the files:\n{e}')
