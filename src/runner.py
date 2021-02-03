import wandb
import numpy as np
import pandas as pd


class Runner:
    # Runner:
    # __init__(self, config):
    # configに渡すもの
    # cvの切り方(train_x, train_y, train_id, n_fold)を受け取ってtrain_idx, val_idxを返すもの。
    # base.BaseModelを継承したモデル
    # wandbのconfig
    def __init__(self, split_method, wandb_config):
        self.split_method = split_method
        self.wandb_config = wandb_config

    def run_cv(self, model, train_x: pd.DataFrame, train_y: pd.DataFrame, train_id=None, test_x=None, n_fold=4,):
        if test_x is not None:
            preds = pd.DataFrame(np.zeros((test_x.shape[0], n_fold)),
                                 columns=list(map(lambda fold: "fold_" + str(fold), range(n_fold))))

        split_idxes = self.split_method(train_x, train_y, train_id, n_fold)
        for i, (train_idx, val_idx) in split_idxes:
            self.wandb_config['name'] = 'fold' + str(i + 1)
            wandb.init(
                self.wandb_config
            )
            tr_x = train_x.iloc[train_idx]
            tr_y = train_y.iloc[train_idx]
            val_x = train_x.iloc[val_idx]
            val_y = train_y.iloc[val_idx]
            model.fit(tr_x, tr_y, val_x, val_y)

            if test_x is not None:
                pred = model.predict(test_x)
                preds.iloc[:, i] = pred

        if test_x is not None:
            return preds


