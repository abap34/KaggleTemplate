import base
import tensorflow as tf
import yaml
import wandb


class SimpleMLPRegressor(base.BaseModel):
    def __init__(self, config):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dropout(config['model']['input_drop_rate']))
        for unit in config['model']['units']:
            model.add(tf.keras.layers.Dense(unit, activation=config['model']['activation']))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(config['model']['drop_rate']))

        self.model.add(tf.keras.layers.Dence(1))

        if config['optimizer']['optimizer'] == "sgd":
            optimizer = tf.keras.optimizers.SGD(
                lr=config['optimizer']['optimizer_lr'], decay=config['optimizer']['decay']
            )
        elif config['optimizer']['optimizer'] == "adam":
            optimizer = tf.keras.optimizers.Adam(
                lr=config['optimizer']['optimizer_lr'], decay=config['optimizer']['decay']
            )
        
        model.compile(optimizer, loss=config['model']['loss'], metrics=config['model']['metrics'])

        self.config = config

    def _fit(self, train_x, train_y, val_x, val_y):
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            **self.config('reduce_lr')
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            **self.config['early_stopping']
        )
        history = self.model.fit(
            train_x,
            train_y,
            epochs=100000,
            validation_data=(val_x, val_y),
            batch_size=self.config['fit']['batch_size'],
            verbose=self.config['fit']['verbose'],
            callbacks=[reduce_lr, early_stopping, tf.keras.callbacks.TerminateOnNaN(), wandb.keras.WandbCallback()],
        )
        return history
    
    def _predict(self, test_x):
        return self.model.predict(test_x)
