import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from pathlib import Path
import numpy as np
import json

class CustomTensorBoard(TensorBoard):
    """
    Extends tensorflow.keras.callbacks TensorBoard

    Custom tensorboard class to make logging of learning rate possible when using
    keras.optimizers.schedules.LearningRateSchedule.
    See https://github.com/tensorflow/tensorflow/pull/37552

    Also logs momemtum for supported optimizers that use momemtum.
    """
    def __init__(self, *args, **kwargs):
        self.dump_history = kwargs.pop("dump_history")
        super().__init__(*args, **kwargs)

    def _collect_learning_rate(self, logs):
        logs = logs or {}
        if hasattr(self.model.optimizer, "lr"):
            lr_schedule = getattr(self.model.optimizer, "lr", None)
            if isinstance(lr_schedule, tf.keras.optimizers.schedules.LearningRateSchedule):
                logs["learning_rate"] = np.float64(tf.keras.backend.get_value(lr_schedule(self.model.optimizer.iterations)))
            else:
                logs.update({"learning_rate": np.float64(tf.keras.backend.eval(self.model.optimizer.lr))})

            # Log momentum if the optimizer has it
            try:
                logs.update({"momentum": np.float64(tf.keras.backend.eval(self.model.optimizer.momentum))})
            except AttributeError:
                pass

            # In Adam, the momentum parameter is called beta_1
            if isinstance(self.model.optimizer, tf.keras.optimizers.Adam):
                logs.update({"adam_beta_1": np.float64(tf.keras.backend.eval(self.model.optimizer.beta_1))})

        return logs

    def on_epoch_end(self, epoch, logs):
        logs = logs or {}
        logs.update(self._collect_learning_rate(logs))
        if self.dump_history:
            history_path = Path(self.log_dir) / "history"
            history_path.mkdir(parents=True, exist_ok=True)
            history_path = str(history_path)
            with open("{}/history_{}.json".format(history_path, epoch), "w") as fi:
                converted_logs = {k: float(v) for k, v in logs.items()}
                json.dump(converted_logs, fi)
        super().on_epoch_end(epoch, logs)

    def on_train_batch_end(self, batch, logs):
        logs = logs or {}
        if isinstance(self.update_freq, int) and batch % self.update_freq == 0:
            logs.update(self._collect_learning_rate(logs))
        super().on_train_batch_end(batch, logs)


class CustomModelCheckpoint(ModelCheckpoint):
    """Extends tensorflow.keras.callbacks.ModelCheckpoint to also save optimizer"""

    def __init__(self, *args, **kwargs):
        # Added arguments
        self.optimizer_to_save = kwargs.pop("optimizer_to_save")
        self.optimizer_filepath = kwargs.pop("optimizer_save_filepath")
        super().__init__(*args, **kwargs)

        Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)

        # If a checkpoint was saved, also save the optimizer
        filepath = str(self.optimizer_filepath).format(epoch=epoch + 1, **logs)
        if self.epochs_since_last_save == 0:
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current == self.best:
                    with open(filepath, "wb") as f:
                        pickle.dump(self.optimizer_to_save, f)
            else:
                with open(filepath, "wb") as f:
                    pickle.dump(self.optimizer_to_save, f)
