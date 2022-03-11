import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from pathlib import Path
import numpy as np
import json
import time
import matplotlib.pyplot as plt
from datetime import datetime
import logging

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


class BenchmarkLogggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, *args, **kwargs):
        # Added arguments
        self.outdir = kwargs.pop("outdir")
        self.steps_per_epoch = kwargs.pop("steps_per_epoch")
        self.batch_size_per_gpu = kwargs.pop("batch_size_per_gpu")
        self.num_gpus = kwargs.pop("num_gpus")
        self.num_devices = kwargs.pop("num_devices")

        if self.num_gpus:
            self.num_devices = self.num_gpus
        elif not self.num_devices:
            # no GPUs, no devices specified, fallback to CPU
            self.num_devices = 1
            logging.warning("No GPUs were found, no devices specified")

        super().__init__(*args, **kwargs)

    def on_train_begin(self, logs=None):
        self.times = []
        self.start_time = tf.timestamp().numpy()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = tf.timestamp().numpy()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(tf.timestamp().numpy() - self.epoch_time_start)

    def plot(self, times):
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Time [s]')
        plt.plot(times, 'o')
        for i in range(len(times)):
            if isinstance(times[i], tf.Tensor):
                j = times[i].numpy()
            else:
                j = times[i]
            if i == 0:
                plt.text(i+0.02, j+0.2, str(round(j, 2)))
            else:
                if isinstance(times[i-1], tf.Tensor):
                    j_prev = times[i-1].numpy()
                else:
                    j_prev = times[i-1]
                plt.text(i+0.02, j+0.2, str(round(j-j_prev, 2)))
        plt.ylim(bottom=0)
        txt = "Time in seconds per epoch. The numbers next to each data point\nshow the difference in seconds compared to the previous epoch."
        plt.title(txt)

        filename = "time_per_epoch_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".png"
        save_path = Path(self.outdir) / filename
        print("Saving plot in {}".format(save_path))
        plt.savefig(save_path)

    def on_train_end(self, logs=None):
        # class methods already do this (eg plot() ) - maybe cast in __init__
        result_path = Path(self.outdir, "result.json")
        stop_time = tf.timestamp().numpy()
        total_time = round(stop_time - self.start_time, 2)

        events_per_epoch = self.num_devices * self.batch_size_per_gpu * self.steps_per_epoch
        throughput_per_epoch = np.repeat(events_per_epoch, len(self.times)) / np.array(
            self.times
        )  # event throughput [1/s]
        mean_throughput = round(np.mean(throughput_per_epoch), 2)

        data = {
            "wl-scores": {
                "throughput_mean": mean_throughput,
            },
            "wl-stats": {
                # may need to generate this array if # epochs is variable...
                "num_epochs": len(self.times),
                "epoch_times": self.times,
                "train_start": self.start_time,
                "train_stop": stop_time,
                "train_time": total_time,
                "GPU": self.num_gpus,
                "CPU": 0 if self.num_gpu else self.num_devices,
                "batch_size_per_device": self.batch_size_per_gpu,
                "batch_total_size": self.batch_size_per_gpu * self.num_devices,
                "steps_per_epoch": self.steps_per_epoch,
                "events_per_epoch": events_per_epoch,
                "throughput_per_epoch": list(throughput_per_epoch),
            },
        }

        print("Saving result to {}".format(result_path.resolve()))
        with result_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        # may not be needed for later versions
        self.plot(self.times)
