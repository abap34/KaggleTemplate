import shutil
import sys
import os


def build_exp_dir(out_path):
    os.makedirs(out_path)
    os.chdir(out_path)
    os.makedirs('data/input')
    shutil.copytree('data/raw')
    os.makedirs('data/output')
    os.makedirs('src/preprocess.py')
    os.makedirs('src/run.py')


if __name__ == '__main__':
    base_path = sys.argv[1]
    out_path = sys.argv[2]
    if base_path == 'new':
        print('generate new template.')
        build_exp_dir(out_path)
    else:
        print(sys.argv)
        shutil.copytree(base_path, out_path)
