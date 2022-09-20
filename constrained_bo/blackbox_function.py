import numpy as np
import tensorflow as tf
from typing import List, Union, Callable
from .utils import is_2Dlistlike
from .constraint import BlackBoxConstraint


class OptimizationProblem:
    """
    Black-box optimisation problem with black-box constraints.
    Each member object of this class is defined by a black-box objective function f(x)
    and one or multiple black-box constraint functions c(x).

    Parameters
    -------
    name: String
        The name of the benchmark problem.

    n_dim: Integer
        The number of input dimensions that are used for the given optimisation problem.

    get_bounds: List
        A method that returns the input domain range.
        This method returns a list of tuples that define the upper and lower bound per input dimension.
        Example: [(0., 1.), (.5, 2.)] define inputs x0 in [0., 1.] and x1 in [.5, 2.]

    point_eval_objective: Callable
        Method that returns the evaluation of the objective at a single point.
        This method must have an input parameter
        Example:



    y_opt: The true

    Returns
    -------
    res: List
        Evaluation result
    """
    def __init__(self,
                 name: str,
                 n_dim: int,
                 bounds: List[tuple],
                 point_eval_objective: Callable,
                 constraints: List[BlackBoxConstraint],
                 y_opt: float,
                 x_opt: Union[tuple, np.array, list]):
        self.name = name
        self.y_opt = y_opt
        self.x_opt = x_opt
        self.n_dim = n_dim
        self.bounds = bounds
        self.constraint_list = constraints
        self._eval_point = point_eval_objective

    def _eval_point(self, X):
        pass

    def __call__(self, X):
        """
        Method that evaluates the objective function at one point

        Parameters
        -------
        X: list or np.ndarray
            Input data

        Returns
        -------
        res: list
            Evaluation result
        """
        # check if multiple points are given
        if not is_2Dlistlike(X):
            X = [X]

        res = []

        for x in X:
            if isinstance(x, tf.Tensor):
                x = tf.squeeze(x)
            res.append(self._eval_point(x))

        if len(res) == 1:
            res = res[0]

        return res
