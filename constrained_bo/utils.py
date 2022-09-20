import gpflow
import tensorflow as tf
import numpy as np
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import math as tfm
from skopt.space import Categorical

def bilog_transform(x):
    """
    The bilog transform that was presented for noisy BO tasks
    in Eriksson & Poloczek, 2020 (https://arxiv.org/abs/2002.08526):
    x' = sign(x) * (1 + |x|)

    :param x: input data
    :return: bilog transform of x
    """
    if isinstance(x, tf.Tensor):
        return tf.math.sign(x) * tf.math.log(1 + tf.math.abs(x))
    return np.sign(x) * np.log(1 + np.abs(x))


def normalise(x):
    """
    A function to normalize data to mean 0 and std=1
    X = (X-mu) / std

    :param x: data to normalize
    :return: normalize data
    """
    mean = tf.math.reduce_mean(x, 0, True)
    std = tf.math.sqrt(tf.math.reduce_variance(x, 0, True))
    return (x - mean) / std

# define own sigmoid transform to avoid numerical instabilities,
# see https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/bijectors/sigmoid.py
# Apparent pros of this alternative
# - 2 fewer subtracts and 1 fewer tf.where, but another add and
#   tf.clip_by_value
# - The answer obviously falls into [lo, hi]
# Apparent cons of this alternative
# - Messing around with clipping and stopping gradients
# - Suppresses any potential severe numerical errors
class mySigmoid(tfb.Sigmoid):
    def _stable_sigmoid(x):
        """A (more) numerically stable sigmoid than `tf.math.sigmoid`."""
        x = tf.convert_to_tensor(x)
        if x.dtype == tf.float64:
            cutoff = -20
        else:
            cutoff = -9
        return tf.where(x < cutoff, tf.exp(x), tf.math.sigmoid(x))

    def _forward(self, x):
        if self._is_standard_sigmoid:
            return self._stable_sigmoid(x)
        lo = tf.convert_to_tensor(self.low)  # Concretize only once
        hi = tf.convert_to_tensor(self.high)
        ans = hi * tf.sigmoid(x) + lo * tf.sigmoid(-x)
        return tfm.clip_by_value_preserve_gradient(ans, lo, hi)


def bounded_hyperparameter(low, high, parameter):
    """
    A function to set transforms that bound the

    :param low: lower bound of the hyperparameter
    :param high: upper bound of the hyperparameter
    :param lengthscale: lengthscale of the hyperparameter
    :return: the new parameter with the given bounds
    """
    bounded_transform = mySigmoid(tf.cast(low, tf.float64), tf.cast(high, tf.float64))
    bounded_parameter = gpflow.Parameter(parameter, transform=bounded_transform, dtype=tf.float64)
    return bounded_parameter


def get_best_feasible(y, constraint_values, constraints):
    """
    Return the best feasible value that was found so far.
    If no feasible value has been returned, the function returns np.nan.

    :param y: the objective function observations
    :param constraint_values: the constraint function observations
    :param constraints: the (true) constraint functions
    :return: the best feasible observation that was made so far
    """
    y = np.asarray(y).flatten()
    mask = np.ones_like(y)
    for i, const in enumerate(constraint_values):
        c_mask = np.asarray(const).flatten() <= constraints[i].rhs
        mask *= c_mask

    # get best feasible value
    feasible_vals = y[np.where(mask)]
    if len(feasible_vals) > 0:
        return np.min(feasible_vals)
    else:
        return np.nan      # if we do not have a feasible value we return nan


def is_listlike(x):
    return isinstance(x, (list, tuple))

def is_2Dlistlike(x):
    return np.all([is_listlike(xi) for xi in x]) \
           or (isinstance(x, np.ndarray) and len(x.shape) == 2)

def get_cont_cat_spaces(space):
    cat_space = []; cont_space = []; cont_idx = []; cat_idx = []
    for i, dim in enumerate(space):
        if isinstance(dim, Categorical):
            cat_space.append(dim.categories)
            cat_idx.append(i)
        else:
            cont_space.append(dim.bounds)
            cont_idx.append(i)
    return cont_space, cat_space, cont_idx, cat_idx
