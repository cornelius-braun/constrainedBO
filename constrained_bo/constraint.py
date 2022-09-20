import numpy as np
import tensorflow as tf
from .utils import is_2Dlistlike

class BlackBoxConstraint:
    """
    Base class to formulate back box constraints.
    Every class object specifies a constraint of the shape 'evaluator' <= 'rhs'.

    Parameters
    ----------
    n_dim : int,
        number of dimensions that we operate in.

    evaluator : function, -> float
        the left-hand side of the constraint. Can be any function that maps to R.

    rhs : float,
        right-hand side of a less-than constraint.
    """
    def __init__(self,
                 n_dim: int,
                 evaluator,
                 rhs: float
                 ):
        self.n_dim = n_dim
        self.evaluator = evaluator
        self.rhs = rhs

    def evaluate(self, X):
        # check if multiple points are given
        if not is_2Dlistlike(X):
            X = [X]

        res = []

        for x in X:
            res.append(self._eval_point(x))

        if len(res) == 1:
            res = res[0]
        else:
            res = np.asarray(res)

        return res

    def __call__(self, X):
        return self.evaluate(X)

    def _eval_point(self, x):
        if isinstance(x, tf.Tensor):
            x = tf.reshape(x, [-1])
        return float(self.evaluator(x))