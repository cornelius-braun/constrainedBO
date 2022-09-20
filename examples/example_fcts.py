import numpy as np
import tensorflow as tf
from skopt import Space
from constrained_bo import BlackBoxConstraint, OptimizationProblem, ConstrainedBO
from timeit import default_timer as timer
from xgboost import XGBClassifier


class G6(OptimizationProblem):
    """
    G6 function: Michalewicz & Schoenauer, 1996
    https://hal.inria.fr/hal-02986407/file/michalewiczSchoenauerECJ96.pdf
    """

    def __init__(self):
        # dimensions
        n_dim = 2

        # objective function
        def _eval_point(x):
            return (x[0] - 10) ** 3 + (x[1] - 20) ** 3

        # constraint functions
        lhs1 = lambda x: -(x[0] - 5) ** 2 - (x[1] - 5) ** 2 + 100
        lhs2 = lambda x: (x[0] - 6) ** 2 + (x[1] - 5) ** 2 - 82.81
        constraint_list = [BlackBoxConstraint(n_dim=n_dim, evaluator=lhs1, rhs=0),
                           BlackBoxConstraint(n_dim=n_dim, evaluator=lhs2, rhs=0)]

        super().__init__(name="G6", n_dim=n_dim, bounds=[(13.5, 14.5), (.5, 1.5)], point_eval_objective=_eval_point,
                         constraints=constraint_list, y_opt=-6961.8139, x_opt=(14.0950, 0.8430))


class XGBoost(OptimizationProblem):
    """
    XGBoost hyperparameter tuning task: Daxberger et al., 2019
    https://arxiv.org/pdf/1907.01329.pdf
    Also used in https://dl.acm.org/doi/pdf/10.1145/3449726.3463136
    """

    def __init__(self):
        from skopt.space import Categorical
        from sklearn import datasets
        from sklearn.preprocessing import MinMaxScaler

        # variables
        self.name = "XGBoost"
        self.y_opt = 0.
        self.n_cv = 5
        self.time = None
        n_cont_dim = 7

        # define data set
        dataset = datasets.load_iris()
        X = dataset.data
        y = dataset.target
        self.X = MinMaxScaler().fit_transform(X)  # normalize X
        self.y = np.array([int(val) for val in y])  # convert y to integer array

        # define dimensions
        self.cat_dims = [
            Categorical([0, 1]),  # booster
            Categorical([3, 100, 5000]),  # n_estimators
            Categorical([1, 10, 15])  # max_depth
        ]
        self.booster_dict = {0: "gbtree", 1: "gblinear"}  # this dict is necessary for compatibility with trieste spaces

        # define time constraint
        def lhs(x):
            if self.time is None:
                self._eval_point(x)  # evaluate objective again to test time constraint
                # raise ValueError("need to evaluate objective before feas")
            res = self.time - 3.
            self.time = None  # flush time after eval to enforce new constraint eval
            return res

        constraint_list = [BlackBoxConstraint(n_dim=n_cont_dim, evaluator=lhs, rhs=0)]

        # define bounds
        bounds = [(0., 109.209690),  # alpha
                  (0.000978, 99.020893),  # lambda
                  (0.046776, 1.),  # colsample_bylevel
                  (0.062528, 1.),  # colsample_bytree
                  (0.000979, 0.995686),  # eta
                  (0.5, 127.041806),  # min_child_weight
                  (0.5, 1.)  # subsample
                  ]
        bounds.extend(self.cat_dims)

        # objective function
        def _eval_point(x):
            from sklearn.model_selection import train_test_split, cross_val_score
            # split data
            if isinstance(x, tf.Tensor):
                x = x.numpy()
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=1_000_000)

            # build a classifier based on selected parameters
            start = timer()
            model = XGBClassifier(alpha=x[0],
                                  reg_lambda=x[1],
                                  colsample_bylevel=x[2],
                                  colsample_bytree=x[3],
                                  eta=x[4],
                                  min_child_weight=x[5],
                                  subsample=x[6],
                                  booster=self.booster_dict[x[7]],
                                  n_estimators=int(x[8]),
                                  max_depth=int(x[9]),
                                  n_jobs=1,
                                  use_label_encoder=False,
                                  verbosity=0)

            # compute scores
            scores = cross_val_score(model, X_train, y_train, cv=self.n_cv, scoring='accuracy')
            acc = 1. - np.mean(scores)
            end = timer()

            # delete model and store time consumed
            del model
            self.time = end - start
            return acc

        super().__init__(name="xgboost", n_dim=n_cont_dim, bounds=bounds, constraints=constraint_list,
                         point_eval_objective=_eval_point, x_opt=np.nan, y_opt=0.0)


class Ackley(OptimizationProblem):
    """
    Ackley function unconstrained
    """

    def __init__(self):
        # define objective and bounds
        n_dim = 20
        bounds = [(-5., 10.) for _ in range(n_dim)]

        def _eval_point(x):
            x = np.asarray(x)
            return 20 + np.e - 20 * np.e ** (-.2 * np.sqrt(np.mean(x ** 2))) - np.e ** (np.mean(np.cos(2 * np.pi * x)))

        super().__init__(name="Ackley", n_dim=n_dim, bounds=bounds, point_eval_objective=_eval_point,
                         x_opt=np.zeros(20), y_opt=0.0, constraints=[])


if __name__ == "__main__":
    # define the problem
    bench = Ackley()
    space = Space(bench.bounds)

    # generate some initial samples
    n_init = 8
    init_X = Space(space).rvs(random_state=1, n_samples=n_init)

    # define optimizer
    optimizer = ConstrainedBO(optimization_problem=bench,
                              init_X=init_X,
                              space=space)

    # run the optimization
    result = optimizer.minimize(n_calls=40,
                                verbose=True)
