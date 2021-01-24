import glob
import pandas as pd
import os
from tqdm import tqdm


def _load_data(full_path):
    print('read:', full_path)
    files = glob.glob(full_path)
    df = pd.DataFrame()
    for file in tqdm(files):
        fe = pd.read_feather(file)
        df = pd.concat([df, fe], axis=1)
    df.sort_index(axis=1, inplace=True)
    return df

def load_data(file_dir):
    train = _load_data(os.path.join(file_dir, 'train/*.feather'))
    test = _load_data(os.path.join(file_dir, 'test/*.feather'))
    
    return train, test



