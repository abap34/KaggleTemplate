from typing import Dict
import base
import tensorflow as tf
import wandb
import pandas as pd
import numpy as np

class SimpleMLPRegressor(base.BaseModel):
    def __init__(self, config : Dict) -> None:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dropout(config['model']['input_drop_rate']))
        for unit in config['model']['units']:
            model.add(tf.keras.layers.Dense(unit, activation=config['model']['activation']))
            if config['model']['batch_norm']:
                model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(config['model']['drop_rate']))

        model.add(tf.keras.layers.Dense(1))

        if config['optimizer']['optimizer'] == "sgd":
            optimizer = tf.keras.optimizers.SGD(
                lr=config['optimizer']['optimizer_lr'], decay=config['optimizer']['decay']
            )
        elif config['optimizer']['optimizer'] == "adam":
            optimizer = tf.keras.optimizers.Adam(
                lr=config['optimizer']['optimizer_lr'], decay=config['optimizer']['decay']
            )
        else:
            err_msg = "ignore optimizer. passed {}".format(config['optimizer']['optimizer'])
            raise ValueError(err_msg)

        
        model.compile(optimizer, loss=config['model']['loss'], metrics=config['model']['metrics'])

        self.model = model

        self.config = config

    def _fit(self, train_x : pd.DataFrame , train_y : pd.DataFrame, val_x : pd.DataFrame, val_y : pd.DataFrame) -> None: 
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            **self.config['reduce_lr']
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            **self.config['early_stopping']
        )
        self.model.fit(
            train_x.values,
            train_y.values,
            epochs=100000,
            validation_data=(val_x.values, val_y.values),
            batch_size=self.config['fit']['batch_size'],
            verbose=self.config['fit']['verbose'],
            callbacks=[reduce_lr, early_stopping,  wandb.keras.WandbCallback()],
        )

    def _predict(self, test_x : pd.DataFrame) -> np.ndaaray:
        return self.model.predict(test_x.values)


class SimpleMLPClassifier(base.BaseModel):
    def __init__(self, config : Dict) -> None:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dropout(config['model']['input_drop_rate']))
        for unit in config['model']['units']:
            model.add(tf.keras.layers.Dense(unit, activation=config['model']['activation']))
            if config['model']['batch_norm']:
                model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(config['model']['drop_rate']))

        model.add(tf.keras.layers.Dense(config['model']['n_class'], activation='softmax'))

        if config['optimizer']['optimizer'] == "sgd":
            optimizer = tf.keras.optimizers.SGD(
                lr=config['optimizer']['optimizer_lr'], decay=config['optimizer']['decay']
            )
        elif config['optimizer']['optimizer'] == "adam":
            optimizer = tf.keras.optimizers.Adam(
                lr=config['optimizer']['optimizer_lr'], decay=config['optimizer']['decay']
            )
        else:
            err_msg = "ignore optimizer. passed {}".format(config['optimizer']['optimizer'])
            raise ValueError(err_msg)

        model.compile(optimizer, loss=config['model']['loss'], metrics=config['model']['metrics'])

        self.model = model

        self.config = config

    def _fit(self, train_x : pd.DataFrame , train_y : pd.DataFrame, val_x : pd.DataFrame, val_y : pd.DataFrame) -> None: 
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            **self.config['reduce_lr']
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            **self.config['early_stopping']
        )
        self.model.fit(
            train_x.values,
            train_y.values,
            epochs=100000,
            validation_data=(val_x.values, val_y.values),
            batch_size=self.config['fit']['batch_size'],
            verbose=self.config['fit']['verbose'],
            callbacks=[reduce_lr, early_stopping,  wandb.keras.WandbCallback()],
        )
    
    def _predict(self, test_x : pd.DataFrame) -> np.ndaaray:
        return self.model.predict(test_x.values)
