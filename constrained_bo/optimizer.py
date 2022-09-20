from typing import Union

import trieste
import gpflow
import numpy as np
import tensorflow as tf
from scipy.optimize import OptimizeResult
from sklearn.preprocessing import MinMaxScaler
from skopt import Space
from trieste.ask_tell_optimization import AskTellOptimizer
from trieste.data import Dataset
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.acquisition.combination import Product
from trieste.models.gpflow import GaussianProcessRegression
from trieste.space import Box, DiscreteSearchSpace, TaggedProductSearchSpace
from .utils import bilog_transform, normalise, bounded_hyperparameter, get_best_feasible, get_cont_cat_spaces
from .blackbox_function import OptimizationProblem

gpflow.config.set_default_float(tf.float64)

# set variables that we will use to access constraint and objective observations
OBJECTIVE = "OBJECTIVE"
CONSTRAINT = "CONSTRAINT"


def create_gp_model(data: Dataset):
    """
    Create a Gaussian Process Regression (GPR) model via GPFlow.
    More information can be found here: https://gpflow.github.io/GPflow/2.5.2/notebooks/basics/regression.html
    and here: https://secondmind-labs.github.io/trieste/0.12.0/notebooks/expected_improvement.html

    :return: an instance of a trieste GPR
    """
    # get gp model
    dim = data.query_points.shape[1]  # input dimension
    model = gpflow.models.GPR((data.query_points, data.observations),
                              kernel=gpflow.kernels.Matern52(lengthscales=np.ones(dim)))

    # initialize hyperparameters
    model.kernel.lengthscales = bounded_hyperparameter(5e-3 * np.ones(dim), 2. * np.ones(dim), 0.5 * np.ones(dim))
    model.kernel.variance = bounded_hyperparameter(5e-2, 20., 1)
    model.likelihood.variance = bounded_hyperparameter(5e-4, 0.2, 5e-3)

    # convert to trieste GPR and optimize hyperparameters
    gpr = GaussianProcessRegression(model)
    gpr.optimize(data)
    return gpr


class ConstrainedBO:
    def __init__(self,
                 optimization_problem: OptimizationProblem,
                 init_X: Union[tf.Tensor, np.ndarray, list],
                 space: Space):
        # store parameters
        self.space = space
        self.init_X = init_X
        self.objective_fct = optimization_problem
        self.constraint_list = optimization_problem.constraint_list
        self.n_constraints = len(self.constraint_list)
        # assert self.n_constraints > 0, "There must be constraints specified!"

        # make sure that we have the constraints in the right format
        for constraint in self.constraint_list:
            assert constraint.rhs == 0

        # define spaces
        self.cont_space, self.cat_space, self.cont_idx, self.cat_idx = get_cont_cat_spaces(space)

        # create search space and box transform
        self.scaler_X = MinMaxScaler(feature_range=(0., 1.))  # create scaler on the space to map the domain to [0, 1]^d
        self.scaler_X.fit(list(map(list, zip(*self.cont_space))))

        # create initial data
        self.dataset = self.observer(self.init_X)

        # define the acquisition; use CWEI = EI * PoF if there are constraints, else normal EI
        if len(self.constraint_list) > 0:
            pof_list = []
            for i, constraint in enumerate(self.constraint_list):
                pof_list.append(
                    trieste.acquisition.ProbabilityOfFeasibility(threshold=constraint.rhs).using(CONSTRAINT + str(i)))
            pof = Product(*pof_list)
            self.acquisition = trieste.acquisition.ExpectedConstrainedImprovement(OBJECTIVE, pof)
        else:
            self.acquisition = trieste.acquisition.ExpectedImprovement()

    def observer(self, query_points: Union[tf.Tensor, np.ndarray, list]):
        """
        Observer function that evaluates the objective and constraints point-wise at a (list of) given query point

        :param query_points: A list or array of points at which to evaluate the objective and constraint functions
        :return: a dictionary of observations for each constraint and the objective function
        """
        Xi = tf.convert_to_tensor(query_points, dtype=tf.double)
        yi = tf.cast(tf.reshape(self.objective_fct(query_points), (-1, 1)), dtype=tf.double)

        # create dictionaries of observations (one per surrogate)
        observation = {OBJECTIVE: Dataset(Xi, yi)}
        for i, constraint in enumerate(self.constraint_list):
            yc = tf.cast(tf.reshape(constraint(query_points), (-1, 1)), dtype=tf.double)
            observation[CONSTRAINT + str(i)] = Dataset(Xi, yc)

        return observation

    def transform_data(self, data: dict):
        """
        Function to normalize the data.
        We follow the approach of Eriksson & Poloczek, 2020 (https://arxiv.org/abs/2002.08526)
        and apply the following transforms:
        1. the input domain X is transformed to [0, 1]^d
        2. the objective function evaluations are normalized to mean=0 and std=1
        3. the constraint function evaluations are normalized using the bi-log transform, which is applied twice

        :param data: the data set that shall be transformed
        :return: the transformed data set
        """
        if len(self.cat_idx) > 0:
            x_sta = self.normalise_mixed_x(data, self.scaler_X)
        else:
            x_sta = self.scaler_X.transform(data[OBJECTIVE].query_points)
        y_sta = normalise(data[OBJECTIVE].observations)
        normalised_data = {OBJECTIVE: Dataset(query_points=x_sta, observations=y_sta)}
        for i, constraint in enumerate(self.constraint_list):
            yc_sta = bilog_transform(bilog_transform(data[CONSTRAINT + str(i)].observations))
            normalised_data[CONSTRAINT + str(i)] = Dataset(x_sta, yc_sta)
        return normalised_data

    def normalise_mixed_x(self, data, scaler):
        """Normalise x and map it to [0, 1] for each continuous dimension, if input domains is mixed.

        Parameters
        -------
        data : trieste.data.Dataset,
            Dictionary that contains the data that shall be normalised.
            We assume that all categorical variables are in the last n columns.
        scaler: sklearn MinMaxScaler,
            scales the data to [0, 1]^d
        cont_idx: list,
            List of indices (columns) that contain continuous variables.
            We assume that all categorical variables are in the last n columns.
        cat_idx: list,
            List of indices (columns) that contain continuous variables.
            We assume that all categorical variables are in the last n columns.
        Returns
        -------
        transformed data:
            np.ndarray with continuous X variables scaled to [0, 1]

        Copyright (c) 2022 Cornelius Braun.
        """
        normalised_cont_x = scaler.transform(data[OBJECTIVE].query_points[:, self.cont_idx[0]:self.cont_idx[-1] + 1])
        cat_x = data[OBJECTIVE].query_points[:, self.cat_idx[0]:]
        return tf.concat([normalised_cont_x, cat_x], axis=-1)

    def minimize(self,
                 n_calls: int = 60,
                 verbose: bool = False):
        """
        Function that executes the Bayesian Optimization loop

        :param n_calls: number of function calls
        :param verbose: verbosity
        :return: best result
        """
        # get initial data and transform it in desired way
        print("Making initial observations at initial inputs")
        dataset = self.observer(self.init_X)
        normalised_observations = self.transform_data(dataset)

        # get the observation space; we scale continuous data to [0, 1] and leave categorical variables untransformed
        normalised_cont_space = Box([0.], [1.]) ** len(self.cont_space)
        transformed_space = [DiscreteSearchSpace(tf.cast(tf.reshape(dim, (-1, 1)), dtype=tf.float64))
                             for dim in self.cat_space]
        transformed_space.insert(0, normalised_cont_space)
        normalised_space = TaggedProductSearchSpace(transformed_space)

        # run optimization loop
        # see for details: https://secondmind-labs.github.io/trieste/0.12.0/notebooks/data_transformation.html
        for step in range(n_calls):
            # create all models; or optimize them after first iteration
            if step == 0:
                models = trieste.utils.map_values(create_gp_model, normalised_observations)
            else:
                for key, model in models.items():
                    model.update(normalised_observations[key])
                    model.optimize(normalised_observations[key])

            # asking for new location at which the objective will be evaluated
            ask_tell = AskTellOptimizer(normalised_space, normalised_observations, models,
                                        acquisition_rule=EfficientGlobalOptimization(self.acquisition), fit_model=False)
            query_point = ask_tell.ask()

            # warp new point back to original space
            query_point_cont = self.scaler_X.inverse_transform(tf.gather(query_point, self.cont_idx, axis=1))
            if len(self.cat_idx) > 0:
                query_point_cat = tf.gather(query_point, self.cat_idx, axis=1)
                query_point = tf.concat([query_point_cont, query_point_cat], axis=-1)
            else:
                query_point = query_point_cont

            # evaluate objective and constraint functions and record data in dataset
            new_observations = self.observer(query_point)
            dataset[OBJECTIVE] += new_observations[OBJECTIVE]
            for i in range(self.n_constraints):
                dataset[CONSTRAINT + str(i)] += new_observations[CONSTRAINT + str(i)]
            normalised_observations = self.transform_data(dataset)  # Normalize the dataset with the new point

            # print current result if we are verbose
            if verbose:
                best_feas = get_best_feasible(y=np.asarray(dataset[OBJECTIVE].observations),
                                              constraint_values=[dataset[CONSTRAINT + str(i)].observations for i in
                                                                 range(self.n_constraints)],
                                              constraints=self.constraint_list)
                print(f"SOLVER: iteration = {step+len(self.init_X)}; best feasible objective val. = {best_feas}")

        # convert result to a scipy result object
        res_yc = [np.asarray(dataset[CONSTRAINT + str(i)].observations).flatten() for i in range(self.n_constraints)]
        res = self.create_result(np.asarray(dataset[OBJECTIVE].query_points),
                                 np.asarray(dataset[OBJECTIVE].observations).flatten(),
                                 res_yc)

        # return best observation
        if verbose:
            best_fun, feasible = get_best_feasible(np.asarray(dataset[OBJECTIVE].observations), res_yc,
                                                   self.constraint_list)
            print("")
            print("SOLVER: finished run!")
            print(f"SOLVER: Function value obtained: {round(best_fun, 5)}; Problem was solved: {feasible}")
            print("")

        return res

    def create_result(self,
                      Xi: Union[tf.Tensor, np.ndarray, list],
                      yi: Union[tf.Tensor, np.ndarray, list],
                      yc: list[Union[tf.Tensor, np.ndarray, list]]):
        """
        Initialize an `OptimizeResult` object.

        Returns
        -------
        res : `OptimizeResult`, scipy object
            OptimizeResult instance with the required information.
        """
        res = OptimizeResult()
        yi = np.asarray(yi)
        if np.ndim(yi) == 2:
            res.log_time = np.ravel(yi[:, 1])
            yi = np.ravel(yi[:, 0])
        best = np.argmin(yi)
        res.x = Xi[best]
        res.fun = yi[best]
        res.func_vals = yi
        res.x_iters = Xi
        res.space = self.space
        res.yc = yc
        return res

