from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class LRFinder(Callback):
    """`Callback` that exponentially adjusts the learning rate after each training batch between `start_lr` and
    `end_lr` for a maximum number of batches: `max_step`. The loss and learning rate are recorded at each step allowing
    visually finding a good learning rate as per https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html via
    the `plot` method.

    A version of this learning rate finder technique is also described under the name 'LR range test' in Leslie Smith's
    paper: https://arxiv.org/pdf/1803.09820.pdf.
    """

    def __init__(
        self,
        start_lr: float = 1e-7,
        end_lr: float = 1e-2,
        max_steps: int = 200,
        smoothing=0.9,
    ):
        super(LRFinder, self).__init__()
        self.start_lr, self.end_lr = start_lr, end_lr
        self.max_steps = max_steps
        self.smoothing = smoothing
        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0
        self.lrs, self.losses = [], []

    def on_train_begin(self, logs=None):
        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0
        self.lrs, self.losses = [], []

    def on_train_batch_begin(self, batch, logs=None):
        self.lr = self.exp_annealing(self.step)
        tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)

    def on_train_batch_end(self, batch, logs=None):
        print("lr:", self.lr)
        print("step", self.step)
        logs = logs or {}
        loss = logs.get("loss")
        step = self.step
        if loss:
            print("loss", loss)
            self.avg_loss = self.smoothing * self.avg_loss + (1 - self.smoothing) * loss
            smooth_loss = self.avg_loss / (1 - self.smoothing ** (self.step + 1))
            self.losses.append(smooth_loss)
            self.lrs.append(self.lr)

            if step == 0 or loss < self.best_loss:
                self.best_loss = loss

            if smooth_loss > 100 * self.best_loss or tf.math.is_nan(smooth_loss):
                self.model.stop_training = True
                print("Loss reached predefined maximum... stopping")
        if step >= self.max_steps:
            print("STOPPING")
            self.model.stop_training = True
        self.step += 1

    def exp_annealing(self, step):
        return self.start_lr * (self.end_lr / self.start_lr) ** (step * 1.0 / self.max_steps)

    def plot(self, save_dir=None, figname="lr_finder.jpg", log_scale=False):
        fig, ax = plt.subplots(1, 1)
        ax.set_ylabel("Loss")
        ax.set_xlabel("Learning Rate")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.0e"))
        ax.plot(self.lrs, self.losses)
        if log_scale:
            ax.set_yscale("log")
        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(str(Path(save_dir) / Path(figname)))
