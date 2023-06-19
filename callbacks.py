import tensorflow as tf
import numpy as np

class PerBatchLrScheduler(tf.keras.callbacks.Callback):
    def __init__(self,model, step_count, epochs, initial_learning_rate=1e-5):
        super(PerBatchLrScheduler, self).__init__()
        self.lr_factor = np.exp(np.log(1e6) / (step_count * epochs))
        self.learning_rate = initial_learning_rate
        self.rates = [self.learning_rate]
        
        
    def on_train_begin(self, logs=None):
        tf.keras.backend.set_value(self.model.optimizer.lr, self.learning_rate)
        
    def on_train_batch_end(self, batch, logs=None):
        new_lr = self.learning_rate * self.lr_factor
        self.learning_rate = new_lr
        self.rates.append(new_lr)
        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
        
class PerBatchLossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))