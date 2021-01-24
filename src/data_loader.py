import glob
from numpy.core.numeric import full
import pandas as pd
import os
from tqdm import tqdm


def _load_data(gl_path):
    files = glob.glob(gl_path)
    df = pd.DataFrame()
    df = pd.concat([pd.read_feather(file) for file in files], axis=1)
    df.sort_index(axis=1, inplace=True)
    return df

def load_data(file_dir):
    train = _load_data(os.path.join(file_dir, 'train/*.feather'))
    test = _load_data(os.path.join(file_dir, 'test/*.feather'))
    
    return train, test



