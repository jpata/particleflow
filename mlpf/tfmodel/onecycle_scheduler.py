import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.python.framework import ops

logging.getLogger("tensorflow").setLevel(logging.ERROR)


class CosineAnnealer:
    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps
        self.n = 0

    def step(self):
        cos = np.cos(np.pi * (self.n / self.steps)) + 1
        self.n += 1
        return self.end + (self.start - self.end) / 2.0 * cos


class OneCycleScheduler(LearningRateSchedule):
    """`LearningRateSchedule` that schedules the learning rate on a 1cycle policy as per Leslie Smith's paper
    (https://arxiv.org/pdf/1803.09820.pdf).

    The implementation adopts additional improvements as per the fastai library:
    https://docs.fast.ai/callbacks.one_cycle.html, where only two phases are used and the adaptation is done using
    cosine annealing. In the warm-up phase the LR increases from `lr_max / div_factor` to `lr_max` and momentum
    decreases from `mom_max` to `mom_min`. In the second phase the LR decreases from `lr_max` to `lr_max / final_div`
    and momemtum from `mom_max` to `mom_min`. By default the phases are not of equal length, with the warm-up phase
    controlled by the parameter `warmup_ratio`.

    NOTE: The momentum is not controlled through this class. This class is intended to be used together with the
    `MomentumOneCycleScheduler` callback defined below.
    """

    def __init__(
        self,
        lr_max,
        steps,
        mom_min=0.85,
        mom_max=0.95,
        warmup_ratio=0.3,
        div_factor=25.0,
        final_div=100000.0,
        name=None,
    ):
        super(OneCycleScheduler, self).__init__()
        lr_min = lr_max / div_factor

        if final_div is None:
            final_lr = lr_max / (div_factor * 1e4)
        else:
            final_lr = lr_max / (final_div)

        phase_1_steps = int(steps * warmup_ratio)
        phase_2_steps = steps - phase_1_steps

        self.lr_max = lr_max
        self.steps = steps
        self.mom_min = mom_min
        self.mom_max = mom_max
        self.warmup_ratio = warmup_ratio
        self.div_factor = div_factor
        self.final_div = final_div
        self.name = name

        phases = [CosineAnnealer(lr_min, lr_max, phase_1_steps), CosineAnnealer(lr_max, final_lr, phase_2_steps)]

        step = 0
        phase = 0
        full_lr_schedule = np.zeros(int(steps))
        for ii in np.arange(np.floor(steps), dtype=int):
            step += 1
            if step >= phase_1_steps:
                phase = 1
            full_lr_schedule[ii] = phases[phase].step()

        self.full_lr_schedule = tf.convert_to_tensor(full_lr_schedule)

    def __call__(self, step):
        with ops.name_scope(self.name or "OneCycleScheduler"):
            return self.full_lr_schedule[tf.cast(step, "int32") - 1]

    def get_config(self):
        return {
            "lr_max": self.lr_max,
            "steps": self.steps,
            "mom_min": self.mom_min,
            "mom_max": self.mom_max,
            "warmup_ratio": self.warmup_ratio,
            "div_factor": self.div_factor,
            "final_div": self.final_div,
            "name": self.name,
        }


class MomentumOneCycleScheduler(Callback):
    """`Callback` that schedules the momentum according to the 1cycle policy as per Leslie Smith's paper
    (https://arxiv.org/pdf/1803.09820.pdf).
    NOTE: This callback only schedules the momentum parameter, not the learning rate. It is intended to be used with the
    KerasOneCycle learning rate scheduler above or similar.
    """

    def __init__(self, steps, mom_min=0.85, mom_max=0.95, warmup_ratio=0.3):
        super(MomentumOneCycleScheduler, self).__init__()

        phase_1_steps = steps * warmup_ratio
        phase_2_steps = steps - phase_1_steps

        self.phase_1_steps = phase_1_steps
        self.phase_2_steps = phase_2_steps
        self.phase = 0
        self.step = 0

        self.phases = [CosineAnnealer(mom_max, mom_min, phase_1_steps), CosineAnnealer(mom_min, mom_max, phase_2_steps)]

    def on_train_begin(self, logs=None):
        self.set_momentum(self.mom_schedule().step())

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if self.step >= self.phase_1_steps:
            self.phase = 1

        self.set_momentum(self.mom_schedule().step())

    def set_momentum(self, mom):
        # In Adam, the momentum parameter is called beta_1
        if hasattr(self.model.optimizer, "beta_1"):
            tf.keras.backend.set_value(self.model.optimizer.beta_1, mom)
        # In SDG, the momentum parameter is called momentum
        elif hasattr(self.model.optimizer, "momentum"):
            tf.keras.backend.set_value(self.model.optimizer.momentum, mom)
        else:
            raise NotImplementedError(
                "Only SGD and Adam are supported by MomentumOneCycleScheduler: {}".format(type(self.model.optimizer))
            )

    def mom_schedule(self):
        return self.phases[self.phase]
