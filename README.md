# Constrained Bayesian Optimization

Implementation of Bayesian Optimization (BO) with Gaussian Process (GP) surrogates based on `GPFlow` and `trieste`.

This code is for constrained global optimization, that attempts to find the minimum value of a black-box
function in a given number of iterations.
More on the algorithm of BO can be found [here](https://ieeexplore.ieee.org/abstract/document/7352306/).

This repository focuses on constrained BO, where the constraints are black-box functions as well.
To learn the unknown feasible region, each constraint is modelled using a separate (GP).

This technique is particularly suited for optimization of functions, situations where the balance
between exploration and exploitation is important.
An unknown constraint could be the run-time of a machine learning of which the hyperparameters are being tuned.

## Quick start
See below for a quick guide on how to use the software.
```
from skopt.space import Space
from constrained_bo import ConstrainedBO
from examples import Ackley

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
result = optimizer.minimize(n_calls=40, verbose=True)
```

