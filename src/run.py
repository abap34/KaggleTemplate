import argparse
from traitlets.traitlets import Undefined
import wandb
import datetime
import pandas as pd
import yaml
from yaml.tokens import ValueToken
import data_loader
from sklearn.model_selection import KFold
import os
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()

    print('[info] received {} as config file. load it...'.format(args.config))

    with open(args.config) as file:
        config = yaml.safe_load(file)

    project_id = config.pop('wandb_project_id')

    data_dir = config['data']['path']
    print('[info] received {} as data dir. load it...'.format(data_dir))

    train, test = data_loader.load_data(data_dir)

    if train.columns.size == 0 or test.columns.size == 0:
        err_msg = 'empty dataframe. Is that a correct path?'
        raise ValueError(err_msg)

    if config['cv']['method'] == 'KFold':
        kf = KFold(
            n_splits=config['cv']['n_fold'],
            shuffle=True,
            random_state=config['cv']['random_state'],
        )
    else:
        err_msg = 'ignonre split fold method. received {}.'.format(
            config['cv']['method']
        )
        raise ValueError(err_msg)

    train_x = train.drop(config['data']['target'], axis=1)
    train_y = train[config['data']['target']]
    if config['task_type'] == 'multiclassification' and config['model-type'] == 'NN':
        train_y = pd.get_dummies(train_y)
    
    if np.sum(train_x.columns != test.columns):
        err_msg = 'Incorrect input. The train column and the test column must match.\n \
            train.columns = {} \n \
            test.columns = {}'.format(
            train.columns, test.columns
        )
        raise ValueError(err_msg)
    preds = np.zeros(test.shape[0])
    val_preds = np.zeros(train_x.shape[0])

    print('[info] start run... \n \
            train_x.shape = {} \n \
            train_y.shape = {} \n \
            test.shape = {}'.format(train_x.shape, train_y.shape, test.shape))
            
    for i, (train_idx, val_idx) in enumerate(kf.split(train_x)):
        run_name = config['model-name'] + '-fold' + str(i + 1)
        wandb.init(
            project=project_id,
            config=config['param'],
            reinit=True,
            name=run_name,
            tags=config['model-type'],
            group=config['run_name'],
        )
        tr_x, val_x = train_x.iloc[train_idx], train_x.iloc[val_idx]
        tr_y, val_y = train_y.iloc[train_idx], train_y.iloc[val_idx]

        if config['model-name'] == 'SimpleMLPRegressor':
            from NNModels import SimpleMLPRegressor

            model = SimpleMLPRegressor(config['param'])
        elif config['model-name'] == 'SimpleMLPClassifier':
            from NNModels import SimpleMLPClassifier
            model = SimpleMLPClassifier(config['param'])
        else:
            err_msg = 'ignonre model types. received {}.'.format(config['model-name'])
            raise ValueError(err_msg)

        model.fit(tr_x, tr_y, val_x, val_y)

        pred = model.predict(test)
        val_pred = model.predict(val_x)
        val_preds[val_idx] = val_pred[:, 0]
        preds += pred[:, 0]

    preds /= config['cv']['n_fold']

    now = datetime.datetime.now()

    outputpath = 'submit/{0:%Y-%m-%d %H:%M:%S}'.format(now)
    os.mkdir(outputpath)

    sample_sub = pd.read_csv('data/raw/sample_submition.csv', header=None)

    sample_sub[:, 1] = preds
    sample_sub.to_csv(outputpath + '/submit.csv', index=False)
