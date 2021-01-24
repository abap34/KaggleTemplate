import argparse
from traitlets.traitlets import Undefined
import wandb
import datetime
import pandas as pd
import yaml
import data_loader
from NNModels import SimpleMLPRegressor
from sklearn.model_selection import KFold
import os
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    print("received {} as config file. load it...".format(args.config))

    with open(args.config) as file:
        config = yaml.safe_load(file)

    data_dir = config["data"]["path"]
    print("received {} as data dir. load it...".format(data_dir))

    train, test = data_loader.load_data(data_dir)

    if config["cv"]["method"] == "KFold":
        kf = KFold(
            n_splits=config["cv"]["n_fold"],
            shuffle=True,
            random_state=config["cv"]["random_state"],
        )
    else:
        err_msg = "ignonre split fold method. received {}.".format(config["cv"]["method"])
        raise ValueError(err_msg)

    train_x = train.drop(config["date"]["target"], axis=1)
    train_y = train[config["date"]["target"]]
    preds = np.zeros(test.shape[0])
    val_preds = np.zeros(train_x.shape[0])
    for train_idx, val_idx in kf.split(train_x):
        tr_x, val_x = train_x.iloc[train_idx], train_x.iloc[val_idx]
        tr_y, val_y = train_y.iloc[train_idx], train_y.iloc[val_idx]

        model = SimpleMLPRegressor(config["param"])

        history = model.fit(tr_x, tr_y, val_x, val_y)

        pred = model.predict(test)
        val_pred = model.predict(val_x)
        val_preds[val_idx] = val_pred
        preds += pred

    preds /= config["date"]["n_fold"]

    # save in wandb.


    wandb.init(project=config.pop("wadb_project_id"), config={config})


    now = datetime.datetime.now()

    outputpath = "submit/{0:%Y-%m-%d %H:%M:%S}".format(now)
    os.mkdir(outputpath)

    sample_sub = pd.read_csv("data/raw/sample_submit.csv", header=False)

    sample_sub[:, 1] = preds

    sample_sub.to_csv(outputpath + "/submit.csv", index=False)

