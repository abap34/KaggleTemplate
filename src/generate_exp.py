import shutil
import sys
import os
import datetime


def build_exp_dir(out_path):
    os.makedirs(out_path)
    shutil.copytree('data/raw', out_path + '/data/input')
    os.chdir(out_path)
    os.makedirs('data/output')
    now = datetime.datetime.now()
    os.makedirs('src')

    s = '# This File is generate by generate_exp.py in ' + now.strftime("%Y/%m/%d %H:%M:%S") + '\n'
    with open('src/preprocess.py', mode='w') as f:
        f.write(s)
    with open('src/run.py', mode='w') as f:
        f.write(s)


if __name__ == '__main__':
    base_path = sys.argv[1]
    out_path = sys.argv[2]
    if base_path == 'new':
        print('generate new template.')
        build_exp_dir(out_path)
    else:
        print(sys.argv)
        shutil.copytree(base_path, out_path)
