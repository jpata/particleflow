import tensorflow as tf
import tensorflow.keras.backend as K


def is_tensor_or_variable(x):
    return tf.is_tensor(x) or isinstance(x, tf.Variable)


class LossFunctionWrapper(tf.keras.losses.Loss):
    """Wraps a loss function in the `Loss` class."""

    def __init__(self, fn, reduction=tf.keras.losses.Reduction.AUTO, name=None, **kwargs):
        """Initializes `LossFunctionWrapper` class.

        Args:
          fn: The loss function to wrap, with signature `fn(y_true, y_pred,
            **kwargs)`.
          reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
            loss. Default value is `AUTO`. `AUTO` indicates that the reduction
            option will be determined by the usage context. For almost all cases
            this defaults to `SUM_OVER_BATCH_SIZE`. When used with
            `tf.distribute.Strategy`, outside of built-in training loops such as
            `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
            will raise an error. Please see this custom training [tutorial](
              https://www.tensorflow.org/tutorials/distribute/custom_training)
            for more details.
          name: (Optional) name for the loss.
          **kwargs: The keyword arguments that are passed on to `fn`.
        """
        super().__init__(reduction=reduction, name=name)
        self.fn = fn
        self._fn_kwargs = kwargs

    def call(self, y_true, y_pred):
        """Invokes the `LossFunctionWrapper` instance.

        Args:
          y_true: Ground truth values.
          y_pred: The predicted values.

        Returns:
          Loss values per sample.
        """
        return self.fn(y_true, y_pred, **self._fn_kwargs)

    def get_config(self):
        config = {}
        for k, v in iter(self._fn_kwargs.items()):
            config[k] = tf.keras.backend.eval(v) if is_tensor_or_variable(v) else v
        base_config = super().get_config()
        return {**base_config, **config}


class SigmoidFocalCrossEntropy(LossFunctionWrapper):
    """Implements the focal loss function.

    Focal loss was first introduced in the RetinaNet paper
    (https://arxiv.org/pdf/1708.02002.pdf). Focal loss is extremely useful for
    classification when you have highly imbalanced classes. It down-weights
    well-classified examples and focuses on hard examples. The loss value is
    much higher for a sample which is misclassified by the classifier as compared
    to the loss value corresponding to a well-classified example. One of the
    best use-cases of focal loss is its usage in object detection where the
    imbalance between the background class and other classes is extremely high.

    Usage:

    >>> fl = tfa.losses.SigmoidFocalCrossEntropy()
    >>> loss = fl(
    ...     y_true = [[1.0], [1.0], [0.0]],y_pred = [[0.97], [0.91], [0.03]])
    >>> loss
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([6.8532745e-06, 1.9097870e-04, 2.0559824e-05],
    dtype=float32)>

    Usage with `tf.keras` API:

    >>> model = tf.keras.Model()
    >>> model.compile('sgd', loss=tfa.losses.SigmoidFocalCrossEntropy())

    Args:
      alpha: balancing factor, default value is 0.25.
      gamma: modulating factor, default value is 2.0.

    Returns:
      Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
          shape as `y_true`; otherwise, it is scalar.

    Raises:
        ValueError: If the shape of `sample_weight` is invalid or value of
          `gamma` is less than zero.
    """

    def __init__(
        self,
        from_logits: bool = False,
        alpha=0.25,
        gamma=2.0,
        reduction: str = tf.keras.losses.Reduction.NONE,
        name: str = "sigmoid_focal_crossentropy",
    ):
        super().__init__(
            sigmoid_focal_crossentropy,
            name=name,
            reduction=reduction,
            from_logits=from_logits,
            alpha=alpha,
            gamma=gamma,
        )


@tf.function
def sigmoid_focal_crossentropy(
    y_true,
    y_pred,
    alpha=0.25,
    gamma=2.0,
    from_logits: bool = False,
) -> tf.Tensor:
    """Implements the focal loss function.

    Focal loss was first introduced in the RetinaNet paper
    (https://arxiv.org/pdf/1708.02002.pdf). Focal loss is extremely useful for
    classification when you have highly imbalanced classes. It down-weights
    well-classified examples and focuses on hard examples. The loss value is
    much higher for a sample which is misclassified by the classifier as compared
    to the loss value corresponding to a well-classified example. One of the
    best use-cases of focal loss is its usage in object detection where the
    imbalance between the background class and other classes is extremely high.

    Args:
        y_true: true targets tensor.
        y_pred: predictions tensor.
        alpha: balancing factor.
        gamma: modulating factor.

    Returns:
        Weighted loss float `Tensor`. If `reduction` is `NONE`,this has the
        same shape as `y_true`; otherwise, it is scalar.
    """
    if gamma and gamma < 0:
        raise ValueError("Value of gamma should be greater than or equal to zero.")

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)

    # Get the cross_entropy for each entry
    ce = K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)

    # If logits are provided then convert the predictions into probabilities
    if from_logits:
        pred_prob = tf.sigmoid(y_pred)
    else:
        pred_prob = y_pred

    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = 1.0
    modulating_factor = 1.0

    if alpha:
        alpha = tf.cast(alpha, dtype=y_true.dtype)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

    if gamma:
        gamma = tf.cast(gamma, dtype=y_true.dtype)
        modulating_factor = tf.pow((1.0 - p_t), gamma)

    # compute the final loss and return
    return tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)
